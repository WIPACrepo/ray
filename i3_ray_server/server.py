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
  - version_policy:    version 3 hardcoded as MODEL_VERSION below
"""

import argparse
import asyncio
import dataclasses as dc
import importlib.metadata
import logging
import os
import signal
import time
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[unresolved-import,import-untyped]
import ray
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ray import serve
from rest_tools.utils import OpenIDAuth
from wipac_dev_tools import from_environment_as_dataclass
from wipac_dev_tools.logging_tools import LoggerLevel


@dc.dataclass(frozen=True)
class EnvConfig:
    """Environment variables."""

    AUTH_AUDIENCE: str = "ray-serve"
    AUTH_OPENID_URL: str = "https://keycloak.icecube.wisc.edu/auth/realms/IceCube"

    CI: bool = False  # github actions sets this to 'true'
    LOG_LEVEL: LoggerLevel = "INFO"


ENV = from_environment_as_dataclass(EnvConfig)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(ENV.LOG_LEVEL)


def _pkg_version(pkg: str) -> str:
    """Return the installed version of a package, or 'unknown' if not found."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "tglauch_classifier"
MODEL_VERSION = "3"
MODEL_PATH = os.path.join(
    "/cvmfs/icecube.opensciencegrid.org/users/briedel/ml/models",
    MODEL_NAME,
    MODEL_VERSION,
    "model.onnx",
)

# Matches config.pbtxt trt_profile_max_shapes batch dimension.
# Triton's max_batch_size (4000) is a queue-level limit; per-inference TRT
# max is 250, so that's the right cap for Ray Serve batching.
BATCH_MAX_SIZE = 250

# Matches config.pbtxt dynamic_batching.max_queue_delay_microseconds = 1000.
BATCH_WAIT_TIMEOUT_S = 0.001

# TensorRT provider options, mirroring config.pbtxt gpu_execution_accelerator.
# Profile shapes match the actual ONNX model: Input-Branch1 is rank-5
# (batch, 10, 10, 60, 16), not rank-4. The cached engine under
# trt_engine_cache_path must be rebuilt after editing these dims.
_TRT_PROVIDER_OPTIONS: dict[str, Any] = {
    "trt_fp16_enable": True,
    "trt_max_workspace_size": 12_884_901_888,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "/tmp/trt_cache",
    "trt_timing_cache_enable": True,
    "trt_context_memory_sharing_enable": True,
    # Profile shapes match config.pbtxt trt_profile_*_shapes.
    "trt_profile_min_shapes": "Input-Branch1:1x10x10x60x16",
    "trt_profile_max_shapes": "Input-Branch1:250x10x10x60x16",
    "trt_profile_opt_shapes": "Input-Branch1:100x10x10x60x16",
}

# Map from V2 datatype strings to numpy dtypes — mirrors the client's _V2_DTYPE_TO_NUMPY.
_V2_DTYPE_TO_NUMPY: dict[str, Any] = {
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "BOOL": np.bool_,
}

# Map from ONNX Runtime input/output type strings to V2 datatype strings.
# ORT sometimes returns "tensor(float)" instead of "tensor(float32)"; both
# map to "FP32" so either form resolves consistently.
_ORT_TYPE_TO_V2: dict[str, str] = {
    "tensor(float16)": "FP16",
    "tensor(float)": "FP32",
    "tensor(float32)": "FP32",
    "tensor(float64)": "FP64",
    "tensor(double)": "FP64",
    "tensor(int8)": "INT8",
    "tensor(int16)": "INT16",
    "tensor(int32)": "INT32",
    "tensor(int64)": "INT64",
    "tensor(uint8)": "UINT8",
    "tensor(uint16)": "UINT16",
    "tensor(uint32)": "UINT32",
    "tensor(uint64)": "UINT64",
    "tensor(bool)": "BOOL",
}


def _ort_shape_to_v2(shape: list) -> list[int]:
    """Convert an ORT shape (with str entries for dynamic dims) to a V2 shape (-1 for dynamic)."""
    # ORT reports symbolic dims as strings (e.g. 'unk__4862'); V2 uses -1.
    return [-1 if isinstance(s, str) else int(s) for s in shape]


# V2 metadata response. Tensor specs (name/shape/datatype) are populated at
# startup from the ONNX session — see TglauchClassifier.__init__. The values
# below are placeholders that get overwritten before any request is served.
_MODEL_METADATA: dict[str, Any] = {
    "name": MODEL_NAME,
    "versions": [MODEL_VERSION],
    "platform": "onnxruntime_onnx",
    "inputs": [{"name": "", "shape": [], "datatype": ""}],
    "outputs": [{"name": "", "shape": [], "datatype": ""}],
    "max_batch_size": BATCH_MAX_SIZE,
}

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
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
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
class TglauchClassifier:
    """Ray Serve deployment serving tglauch_classifier via ONNX Runtime + TRT."""

    def __init__(self):
        providers = [
            ("TensorrtExecutionProvider", _TRT_PROVIDER_OPTIONS),
            "CUDAExecutionProvider",
            # CPU fallback — remove if GPU is guaranteed.
            "CPUExecutionProvider",
        ]
        sess_options = ort.SessionOptions()
        # Matches config.pbtxt optimization.graph.level = 1.
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        # Parallelism is handled by Ray Serve replicas (4 × 1 GPU each),
        # not by threading within a session. Keeping both thread counts at 1
        # avoids TRT context thread-safety issues.
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            MODEL_PATH, sess_options=sess_options, providers=providers
        )

        # Resolve all tensor metadata from the actual ONNX model at startup so
        # the V2 metadata endpoint never drifts from what the model accepts /
        # returns. Without this, hardcoded shapes/dtypes in _MODEL_METADATA can
        # silently lie about the model (e.g. wrong rank, wrong output dtype).
        ort_in = self._session.get_inputs()[0]
        ort_out = self._session.get_outputs()[0]
        self._input_name: str = ort_in.name
        self._output_name: str = ort_out.name
        self._output_datatype: str = _ORT_TYPE_TO_V2[ort_out.type]

        _MODEL_METADATA["inputs"][0]["name"] = self._input_name
        _MODEL_METADATA["inputs"][0]["shape"] = _ort_shape_to_v2(ort_in.shape)
        _MODEL_METADATA["inputs"][0]["datatype"] = _ORT_TYPE_TO_V2[ort_in.type]
        _MODEL_METADATA["outputs"][0]["name"] = self._output_name
        _MODEL_METADATA["outputs"][0]["shape"] = _ort_shape_to_v2(ort_out.shape)
        _MODEL_METADATA["outputs"][0]["datatype"] = self._output_datatype

        LOGGER.info(
            f"TglauchClassifier ready  "
            f"model={MODEL_NAME!r} version={MODEL_VERSION}  "
            f"auth={'disabled (CI)' if ENV.CI else 'enabled'}"
        )
        # Update 'i3-ml-suite' below to match the actual deployed package name.
        LOGGER.info(
            f"versions:  "
            f"server={_pkg_version('i3-ml-suite')}  "
            f"onnxruntime={_pkg_version('onnxruntime')}  "
            f"ray={_pkg_version('ray')}"
        )
        LOGGER.info(f"model path: {MODEL_PATH}")
        LOGGER.info(
            f"ONNX tensors:  "
            f"input={self._input_name!r} shape={_MODEL_METADATA['inputs'][0]['shape']} "
            f"dtype={_MODEL_METADATA['inputs'][0]['datatype']}  |  "
            f"output={self._output_name!r} shape={_MODEL_METADATA['outputs'][0]['shape']} "
            f"dtype={self._output_datatype}"
        )
        LOGGER.info(
            f"batching:  max_size={BATCH_MAX_SIZE}  "
            f"wait_timeout={BATCH_WAIT_TIMEOUT_S}s"
        )

    # -----------------------------------------------------------------------
    # Health endpoints
    # -----------------------------------------------------------------------

    @app.get("/v2/health/live")
    async def health_live(
        self,
        # NOTE: no auth because this endpoint is public to readiness probes
    ):
        """Liveness probe — server process is running."""
        LOGGER.debug("health/live")
        return {}

    @app.get("/v2/health/ready")
    async def health_ready(
        self,
        # NOTE: no auth because this endpoint is public to readiness probes
    ):
        """Readiness probe — server is ready to accept requests."""
        LOGGER.debug("health/ready")
        return {}

    @app.get(f"/v2/models/{MODEL_NAME}/ready")
    @app.get(f"/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}/ready")
    async def model_ready(
        self,
        _claims: dict = Depends(Auth.require_auth),
    ):
        """Model readiness probe."""
        LOGGER.debug(f"model_ready  model={MODEL_NAME!r} version={MODEL_VERSION}")
        return {}

    # -----------------------------------------------------------------------
    # Metadata endpoints
    # -----------------------------------------------------------------------

    @app.get(f"/v2/models/{MODEL_NAME}")
    @app.get(f"/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}")
    async def model_metadata(
        self,
        _claims: dict = Depends(Auth.require_auth),
    ):
        """Return V2 model metadata including input/output tensor specs."""
        # _MODEL_METADATA is fully populated in __init__ from the ONNX session,
        # so it can be returned directly.
        LOGGER.debug(f"model_metadata  model={MODEL_NAME!r} version={MODEL_VERSION}")
        return _MODEL_METADATA

    # -----------------------------------------------------------------------
    # Inference endpoints
    # -----------------------------------------------------------------------

    @app.post(f"/v2/models/{MODEL_NAME}/infer")
    @app.post(f"/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}/infer")
    async def infer(
        self,
        http_request: Request,
        _claims: dict = Depends(Auth.require_auth),
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

        dtype = _V2_DTYPE_TO_NUMPY.get(tensor.get("datatype", "FP16"), np.float16)
        input_array = np.array(tensor["data"], dtype=dtype).reshape(tensor["shape"])

        t0 = time.monotonic()
        try:
            result = await self._run_inference(input_array)
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
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "outputs": [
                {
                    "name": self._output_name,
                    # Use the dtype resolved from the model session, not a
                    # hardcoded string — otherwise the client downcasts.
                    "datatype": self._output_datatype,
                    "shape": list(result.shape),
                    "data": result.flatten().tolist(),
                }
            ],
        }

    @serve.batch(
        max_batch_size=BATCH_MAX_SIZE,
        batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S,
    )
    async def _run_inference(self, input_arrays: list[np.ndarray]) -> list[np.ndarray]:
        """Run batched ONNX inference.

        Ray Serve collects individual infer() calls into a batch here,
        replacing Triton's dynamic_batching behavior.
        """
        # Stack individual requests into a single batched array.
        batched = np.concatenate(input_arrays, axis=0)

        LOGGER.debug(
            f"_run_inference: batching {len(input_arrays)} request(s) → shape {batched.shape}"
        )

        # session.run is synchronous and GPU-bound; offload it to a thread
        # executor so the Ray Serve event loop is not blocked during inference.
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._session.run(
                [self._output_name],
                {self._input_name: batched},
            ),
        )
        batched_result = np.asarray(outputs[0])
        LOGGER.debug(
            f"_run_inference: complete  "
            f"out_shape={batched_result.shape}  "
            f"{(time.monotonic() - t0) * 1000:.1f}ms"
        )

        # Split the batched output back into per-request arrays.
        # np.split with an empty index list (single request case) correctly
        # returns [batched_result] unchanged.
        sizes = [arr.shape[0] for arr in input_arrays]
        split_indices: list[int] = np.cumsum(sizes)[:-1].tolist()
        return list(
            np.split(batched_result, split_indices)  # ty: ignore[no-matching-overload]
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ray Serve V2 server for tglauch_classifier"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ray.init()
    serve.start(http_options={"host": args.host, "port": args.port})
    # fmt: off
    serve.run(TglauchClassifier.bind())  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    # fmt: on
    # Block until SIGINT/SIGTERM. serve.run_until_interrupted() does not exist
    # in current Ray releases; signal.pause() is the portable equivalent.
    signal.pause()

# Assign TglauchClassifier application to variable for ray serve exectution in kubernetes
# else run directly
if "KUBERNETES_SERVICE_HOST" in os.environ:
    model = TglauchClassifier.bind()  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
