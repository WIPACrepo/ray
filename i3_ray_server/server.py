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
import os
import signal
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[unresolved-import,import-untyped]
import ray
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ray import serve
from rest_tools.server.auth import OpenIDAuth

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
_TRT_PROVIDER_OPTIONS: dict[str, Any] = {
    "trt_fp16_enable": True,
    "trt_max_workspace_size": 12_884_901_888,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "/tmp/trt_cache",
    "trt_timing_cache_enable": True,
    "trt_context_memory_sharing_enable": True,
    # Profile shapes match config.pbtxt trt_profile_*_shapes.
    "trt_profile_min_shapes": "Input-Branch1:1x10x60x16",
    "trt_profile_max_shapes": "Input-Branch1:250x10x60x16",
    "trt_profile_opt_shapes": "Input-Branch1:100x10x60x16",
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

# V2 metadata response — static, matches model inputs/outputs.
_MODEL_METADATA: dict[str, Any] = {
    "name": MODEL_NAME,
    "versions": [MODEL_VERSION],
    "platform": "onnxruntime_onnx",
    "inputs": [
        {
            "name": "Input-Branch1",
            "datatype": "FP16",
            # -1 indicates dynamic batch axis.
            "shape": [-1, 10, 60, 16],
        }
    ],
    "outputs": [
        {
            "name": "output",  # TODO: confirm output tensor name from model
            "datatype": "FP16",
            "shape": [-1],  # TODO: confirm output shape from model
        }
    ],
    "max_batch_size": BATCH_MAX_SIZE,
}

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

# OpenIDAuth is instantiated once at module load. It fetches Keycloak's
# .well-known/openid-configuration and caches public keys for offline JWT
# validation. Set AUTH_OPENID_URL + AUTH_AUDIENCE in the environment; leave
# them empty (or set CI=true) to skip auth in tests.
_AUTH_OPENID_URL = os.getenv("AUTH_OPENID_URL", "")
_AUTH_AUDIENCE = os.getenv("AUTH_AUDIENCE", "")
_CI = os.getenv("CI", "").lower() == "true"
_openid_auth: OpenIDAuth | None = (
    OpenIDAuth(_AUTH_OPENID_URL, audience=_AUTH_AUDIENCE)
    if _AUTH_OPENID_URL and not _CI
    else None
)
_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """FastAPI dependency that validates the Keycloak Bearer token.

    Returns the decoded JWT claims on success.
    Raises HTTP 403 on missing or invalid token.

    TODO: add role check here, e.g.:
        roles = claims.get("realm_access", {}).get("roles", [])
        if "my-role" not in roles:
            raise HTTPException(status_code=403, detail="insufficient role")
    """
    if _openid_auth is None:
        # Auth disabled (CI=true or no AUTH_OPENID_URL set).
        return {}
    if credentials is None:
        raise HTTPException(status_code=403, detail="missing authorization header")
    try:
        return _openid_auth.validate(credentials.credentials)
    except Exception as exc:
        raise HTTPException(status_code=403, detail="invalid token") from exc


# ---------------------------------------------------------------------------
# FastAPI app (shared across all replicas via Ray Serve ingress)
# ---------------------------------------------------------------------------

app = FastAPI()


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


@serve.deployment(
    # Matches config.pbtxt instance_group count=4, kind=KIND_GPU.
    num_replicas=4,
    ray_actor_options={"num_gpus": 1},
)
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
        # Resolve the actual output tensor name from the model at startup
        # and write it into _MODEL_METADATA so metadata responses are correct.
        self._output_name: str = self._session.get_outputs()[0].name
        _MODEL_METADATA["outputs"][0]["name"] = self._output_name

    # -----------------------------------------------------------------------
    # Health endpoints
    # -----------------------------------------------------------------------

    @app.get("/v2/health/live")
    async def health_live(self):
        """Liveness probe — server process is running."""
        return {}

    @app.get("/v2/health/ready")
    async def health_ready(self):
        """Readiness probe — server is ready to accept requests."""
        return {}

    @app.get(f"/v2/models/{MODEL_NAME}/ready")
    @app.get(f"/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}/ready")
    async def model_ready(self):
        """Model readiness probe."""
        return {}

    # -----------------------------------------------------------------------
    # Metadata endpoints
    # -----------------------------------------------------------------------

    @app.get(f"/v2/models/{MODEL_NAME}")
    @app.get(f"/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}")
    async def model_metadata(self, _claims: dict = Depends(require_auth)):
        """Return V2 model metadata including input/output tensor specs."""
        # _MODEL_METADATA is fully populated in __init__ (output name resolved
        # from the session), so it can be returned directly.
        return _MODEL_METADATA

    # -----------------------------------------------------------------------
    # Inference endpoints
    # -----------------------------------------------------------------------

    @app.post(f"/v2/models/{MODEL_NAME}/infer")
    @app.post(f"/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}/infer")
    async def infer(self, http_request: Request, _claims: dict = Depends(require_auth)):
        """Accept a V2 infer request and return a V2 infer response."""
        request = await http_request.json()
        request_id = request.get("id", "")
        inputs = request.get("inputs", [])

        if not inputs:
            raise HTTPException(status_code=400, detail="No inputs provided")
        # Currently only supporting single-input models.
        if len(inputs) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 1 input tensor, got {len(inputs)}",
            )

        tensor = inputs[0]
        dtype = _V2_DTYPE_TO_NUMPY.get(tensor.get("datatype", "FP16"), np.float16)
        input_array = np.array(tensor["data"], dtype=dtype).reshape(tensor["shape"])

        result = await self._run_inference(input_array)

        return {
            "id": request_id,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "outputs": [
                {
                    "name": self._output_name,
                    "datatype": "FP16",
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

        # session.run is synchronous and GPU-bound; offload it to a thread
        # executor so the Ray Serve event loop is not blocked during inference.
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._session.run(
                [self._output_name],
                {"Input-Branch1": batched},
            ),
        )
        batched_result = outputs[0]

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
