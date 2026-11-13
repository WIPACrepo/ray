"""
Ray Serve V2 inference server for tglauch_classifier.

Replaces the Triton server. Implements the KServe Open Inference Protocol
(V2) HTTP API so existing clients using ray_serve.py work unchanged.

Triton config.pbtxt equivalents:
    - platform:          onnxruntime + TensorRT execution provider
    - instance_group:    num_replicas=4, num_gpus=1 per replica
    - max_batch_size:    4000 (Triton total); TRT profile max is 250 per call,
                         so Ray Serve batching is capped at 250
    - dynamic_batching:  @serve.batch with batch_wait_timeout_s=0.001 (1ms)
    - version_policy:    version 3 hardcoded as self.model_version below
"""

import asyncio
import dataclasses as dc
import importlib.metadata
import logging
import os
import time
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from ray import serve  # type: ignore[unresolved-import,import-untyped]
from ray.serve.config import BackpressureConfig  # type: ignore[unresolved-import,import-untyped]
from rest_tools.utils import OpenIDAuth
from wipac_dev_tools import from_environment_as_dataclass
from wipac_dev_tools.logging_tools import LoggerLevel

from inference_backend import *


@dc.dataclass(frozen=True)
class EnvConfig:
    """Environment variables."""

    AUTH_AUDIENCE: str = "ray-serve"
    AUTH_OPENID_URL: str = "https://keycloak.icecube.wisc.edu/auth/realms/IceCube"

    CI: bool = False  # github actions sets this to 'true'
    LOG_LEVEL: LoggerLevel = "INFO"


ENV = from_environment_as_dataclass(EnvConfig)  # pyright: ignore
LOGGER = logging.getLogger("ray")
LOGGER.setLevel(ENV.LOG_LEVEL)


def _pkg_version(pkg: str) -> str:
    """Return the installed version of a package, or 'unknown' if not found."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class Auth:
    """
    A wrapper around OpenIDAuth that validates the Keycloak Bearer token.

    OpenIDAuth is instantiated once at module load. It fetches Keycloak's
    .well-known/openid-configuration and caches public keys for offline JWT
    validation.
    """

    _openid_auth: OpenIDAuth | None = (
        None
        if ENV.CI  # no auth checks in CI
        else OpenIDAuth(ENV.AUTH_OPENID_URL, audience=ENV.AUTH_AUDIENCE)
    )
    _bearer = HTTPBearer(auto_error=False)

    _AUTH_ROLE = "admin"  # we can update this and/or add more roles later (parametrized by caller)

    @staticmethod
    async def require_auth(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> dict[str, Any]:
        """FastAPI dependency that validates the Keycloak Bearer token.

        Returns the decoded JWT claims on success.
        Raises HTTP 403 on missing token, invalid token, or missing 'admin' role.
        """
        if Auth._openid_auth is None:
            return {}  # Auth disabled

        if credentials is None:
            LOGGER.warning("auth rejected: missing authorization header")
            raise HTTPException(status_code=403, detail="missing authorization header")

        try:
            claims = Auth._openid_auth.validate(credentials.credentials)
        except Exception as exc:
            LOGGER.warning(f"auth rejected: invalid token: {exc}")
            raise HTTPException(status_code=403, detail="invalid token") from exc

        # Keycloak client roles live under resource_access.<client_id>.roles.
        roles: list[str] = (
            claims.get("resource_access", {})
            .get(ENV.AUTH_AUDIENCE, {})
            .get("roles", [])
        )

        if Auth._AUTH_ROLE not in roles:
            LOGGER.warning(
                f"auth rejected: missing role {Auth._AUTH_ROLE!r}  present={roles}"
            )
            raise HTTPException(status_code=403, detail="insufficient role")
        return claims


# ---------------------------------------------------------------------------
# FastAPI app (shared across all replicas via Ray Serve ingress)
# ---------------------------------------------------------------------------

app = FastAPI()


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


@serve.deployment()
@serve.ingress(app)
class GenericServe:
    """Ray Serve deployment serving an InferenceBackend"""

    def __init__(self, config: dict):
        self._build(BackendConfig(**config))

        LOGGER.info(
            f"Serve deployment ready  "
            f"model={self.backend.model_name!r} version={self.backend.model_version}  "
            f"auth={'disabled (CI)' if ENV.CI else 'enabled'}"
        )
        LOGGER.info(
            f"versions:  "
            f"server={_pkg_version('i3-ml-suite')}  "
            f"onnxruntime={_pkg_version('onnxruntime')}  "
            f"ray={_pkg_version('ray')}"
        )
        LOGGER.info(f"model path: {self.backend.model_path}")
        LOGGER.info(
            f"ONNX tensors:  "
            f"inputs: \n{self.backend.model_metadata['inputs']}\n"
            f"outputs: \n{self.backend.model_metadata['outputs']}"
        )
        LOGGER.info(
            f"batching:  max_size={self.backend.options_get('batch_max_size')}  "
            f"wait_timeout={self.backend.options_get('batch_wait_timeout_s')}s"
        )

    def _build(self, config: BackendConfig):
        backend = create_backend(config)
        backend.load()

        self.config, self.backend = config, backend

    # -----------------------------------------------------------------------
    # Health endpoints
    # -----------------------------------------------------------------------
    @app.get("/ready")
    async def model_ready(
        self,
        _claims: Annotated[dict, Depends(Auth.require_auth)],
    ):
        """Model readiness probe."""
        LOGGER.debug(
            f"model_ready  model={self.backend.model_name!r} version={self.backend.model_version}"
        )
        return {}

    # -----------------------------------------------------------------------
    # Metadata endpoints
    # -----------------------------------------------------------------------

    @app.get("/")
    async def model_metadata(
        self,
        _claims: Annotated[dict, Depends(Auth.require_auth)],
    ):
        """Return V2 model metadata including input/output tensor specs."""
        # _MODEL_METADATA is fully populated in __init__ from the ONNX session,
        # so it can be returned directly.
        LOGGER.debug(
            f"model_metadata  model={self.backend.model_name!r} version={self.backend.model_version}"
        )
        return self.backend.model_metadata

    # -----------------------------------------------------------------------
    # Inference endpoints
    # -----------------------------------------------------------------------

    @app.get("/warm")
    async def warm(
        self,
        _claims: Annotated[dict, Depends(Auth.require_auth)],
    ):
        """Warm Exection Provider by running inference with zeroed tensor input array"""
        try:
            result = await self.backend.warm()
        except Exception as exc:
            LOGGER.error(f"warm failed {exc}")
            raise

        return {
            "id": "warm",
            "model_name": self.backend.model_name,
            "model_version": self.backend.model_version,
            "outputs": [
                {
                    "name": self.backend.model_metadata["outputs"]["name"],
                    "datatype": self.backend.model_metadata["outputs"]["datatype"],
                    "shape": self.backend.model_metadata["outputs"]["shape"],
                    "data": result.flatten().tolist(),
                }
            ],
        }

    @app.post("/infer")
    async def infer(
        self,
        http_request: Request,
        _claims: Annotated[dict, Depends(Auth.require_auth)],
    ):
        """Accept a V2 infer request and return a V2 infer response."""
        request = await http_request.json()
        request_id = request.get("id", "")
        inputs = request.get("inputs", [])

        if not inputs:
            LOGGER.warning(f"infer rejected: no inputs  id={request_id[:8]!r}")
            raise HTTPException(status_code=400, detail="No inputs provided")
        # Currently only supporting single-input models.
        if len(inputs) != 1:
            LOGGER.warning(
                f"infer rejected: expected 1 input tensor, got {len(inputs)}  "
                f"id={request_id[:8]!r}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Expected 1 input tensor, got {len(inputs)}",
            )
        tensor = inputs[0]
        input_shape = tensor.get("shape", [])
        LOGGER.debug(f"infer request  id={request_id[:8]!r} shape={input_shape}")

        dtype = InferenceBackend._V2_DTYPE_TO_NUMPY.get(
            tensor.get("datatype", "FP16"), np.float16
        )
        input_array = np.array(tensor["data"], dtype=dtype).reshape(tensor["shape"])

        t0 = time.monotonic()
        try:
            result = await self._batch_inference(input_array)
        except Exception as exc:
            LOGGER.error(
                f"infer failed  id={request_id[:8]!r} shape={input_shape}: {exc}"
            )
            raise

        LOGGER.info(
            f"infer  id={request_id[:8]!r}  "
            f"in={input_shape}  out={list(result.shape)}  "
            f"{(time.monotonic() - t0) * 1000:.1f}ms"
        )

        return {
            "id": request_id,
            "model_name": self.backend.model_name,
            "model_version": self.backend.model_version,
            "outputs": [
                {
                    "name": self.backend.model_metadata["outputs"][0]["name"],
                    "datatype": self.backend.model_metadata["outputs"][0]["name"],
                    "shape": list(result.shape),
                    "data": result.flatten().tolist(),
                }
            ],
        }

    @serve.batch(
        batch_size_fn=lambda arrays: sum(len(a) for a in arrays),
    )
    async def _batch_inference(
        self, input_arrays: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Ray Serve collects individual infer() calls into a batch here
        reconfigure is ran to set appropriate batch size and timeouts on infer from user_config in ServeConfig
        """
        # Stack individual requests into a single batched array.
        batched = np.concatenate(input_arrays, axis=0)

        LOGGER.debug(
            f"_run_inference: batching {len(input_arrays)} request(s) → shape {batched.shape}"
        )

        results = await self.backend.infer(batched)

        return results

    async def reconfigure(self, user_config: dict):
        """Set the max_batch_size and batch_wait_timeout_s via user_config if available"""
        max_batch_size = user_config.get("max_batch_size", 1)
        batch_wait_timeout_s = user_config.get("batch_wait_timeout_s", 0.001)

        self._batch_inference.set_max_batch_size(max_batch_size)  # pyright: ignore[reportAttributeAccessIssue]
        self._batch_inference.set_batch_wait_timeout_s(batch_wait_timeout_s)  # pyright: ignore[reportAttributeAccessIssue]


# model = GenericServe.bind(args)  # type: ignore[attr-defined]
def build(args: dict):
    # Validate at deploy time so bad config fails here, not in the replica
    # BackendConfig(**args)
    return GenericServe.bind(args)
