import asyncio
import dataclasses as dc
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
# from pydantic import BaseModel

LOGGER = logging.getLogger("ray")


@dc.dataclass
class BackendConfig:
    """BackendConfig: dataclass for initializing a InferenceBackend with args from a ServeConfig

    Attributes:
        runtime: str
            Inference backend to use
        model_repo: str
            Base repository to look for a model (ex: /cvmfs/icecube.wisc.edu/...)
        model_name: str
            Model name
        model_version: str
            Model version
        model_path: str | None = None
            Can be a specific path, or leave none to automatically build from model_repo, model_name, model_version
        options: dict[str, Any]
            Dict containing runtime specific configuration for a InferenceBackend
    """

    runtime: str
    model_repo: str
    model_name: str
    model_version: str
    model_path: str | None = None
    options: dict[str, Any] = dc.field(default_factory=dict)


class InferenceBackend(ABC):
    # Map from Kserve V2 datatype strings to numpy dtypes — mirrors the client's _V2_DTYPE_TO_NUMPY.
    _V2_DTYPE_TO_NUMPY: ClassVar[dict[str, Any]] = {
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

    def __init__(self, config: BackendConfig):
        self._config = config
        self.model_name = config.model_name
        self.model_version = config.model_version
        self.model_path = self._get_model_path()
        self.model_metadata: dict[str, Any] = {
            "name": self._config.model_name,
            "versions": [self._config.model_version],
            "platform": self._config.runtime,
            "inputs": [{"name": "", "shape": [], "datatype": ""}],
            "outputs": [{"name": "", "shape": [], "datatype": ""}],
        }

    def _get_model_path(self) -> str:
        return self._config.model_path or os.path.join(
            self._config.model_repo,
            self._config.model_name,
            self._config.model_version,
            "model.onnx",
        )

    def options_get(self, key: str, default: Any = None) -> Any | None:
        return self._config.options.get("key", default)

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    async def infer(self, inputs: list[npt.NDArray[Any]]) -> list[npt.NDArray[Any]]: ...

    @abstractmethod
    async def warm(self) -> Any: ...

    def close(self) -> None: ...


_REGISTRY: dict[str, type[InferenceBackend]] = {}


def register_backend(name: str):
    """Class decorator for registering as an extension of InferenceBackend"""

    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def create_backend(config: BackendConfig) -> InferenceBackend:
    try:
        return _REGISTRY[config.runtime](config)
    except KeyError:
        raise ValueError(f"Unknown runtime {config.runtime!r}; have {list(_REGISTRY)}")


@register_backend("onnx_onnxv1")
class OnnxBackend(InferenceBackend):
    # Map from ONNX Runtime input/output type strings to V2 datatype strings.
    # ORT sometimes returns "tensor(float)" instead of "tensor(float32)"; both
    # map to "FP32" so either form resolves consistently.
    _ORT_TYPE_TO_V2: ClassVar[dict[str, str]] = {
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

    def load(self):
        import onnxruntime as ort

        ort.preload_dlls(directory="")

        sess_options = ort.SessionOptions()

        providers = self._configure_providers()

        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_profiling = True
        # Parallelism is handled by Ray Serve replicas (4 × 1 GPU each),
        # not by threading within a session. Keeping both thread counts at 1
        # avoids TRT context thread-safety issues.
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1

        self._session = ort.InferenceSession(
            self._get_model_path(),
            sess_options=sess_options,
            providers=providers,
        )

        # Resolve all tensor metadata from the actual ONNX model at startup so
        # the V2 metadata endpoint never drifts from what the model accepts /
        # returns. Without this, hardcoded shapes/dtypes in model_metadata can
        # silently lie about the model (e.g. wrong rank, wrong output dtype).
        ort_in = self._session.get_inputs()[0]
        ort_out = self._session.get_outputs()[0]
        self._input_name = ort_in.name
        self._output_name = ort_out.name

        self.model_metadata["inputs"]["name"] = self._input_name
        self.model_metadata["inputs"]["shape"] = self._ort_shape_to_v2(ort_in.shape)
        self.model_metadata["inputs"]["datatype"] = OnnxBackend._ORT_TYPE_TO_V2[
            ort_in.type
        ]

        self.model_metadata["outputs"]["name"] = self._output_name
        self.model_metadata["outputs"]["shape"] = self._ort_shape_to_v2(ort_out.shape)
        self.model_metadata["outputs"]["datatype"] = OnnxBackend._ORT_TYPE_TO_V2[
            ort_out.type
        ]

    async def warm(self) -> list[npt.NDArray[Any]]:
        """Warm Exection Provider by running inference with zeroed tensor input array"""
        # Convert Onnx Dtype to Numpy; Set shape dynamic size to 1
        dtype: npt.DTypeLike = InferenceBackend._V2_DTYPE_TO_NUMPY[
            self.model_metadata["inputs"]["datatype"]
        ]
        shape = self.model_metadata["inputs"]["shape"]
        shape[0] = 1
        input_array = np.zeros_like(shape, dtype=dtype)

        t0 = time.monotonic()
        try:
            result = await self.infer([input_array])
        except Exception as exc:
            LOGGER.error(f"infer failed id=prewarm, dtype={dtype} shape={shape}: {exc}")
            raise

        LOGGER.info(
            f"infer "
            f"in={shape}  out={list(result[0].shape)}  "
            f"{(time.monotonic() - t0) * 1000:.1f}ms"
        )

        return result

    async def infer(self, inputs: list[npt.NDArray[Any]]) -> list[npt.NDArray[Any]]:
        """Run batched ONNX inference.

        Ray Serve collects individual infer() calls into a batch here,
        replacing Triton's dynamic_batching behavior.
        """
        # session.run is synchronous and GPU-bound; offload it to a thread
        # executor so the Ray Serve event loop is not blocked during inference.
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._session.run(
                [self._output_name],
                {self._input_name: inputs},
            ),
        )
        result = np.asarray(outputs[0])
        LOGGER.debug(
            f"_run_inference: complete  "
            f"out_shape={result.shape}  "
            f"{(time.monotonic() - t0) * 1000:.1f}ms"
        )

        # Split the batched output back into per-request arrays.
        # np.split with an empty index list (single request case) correctly
        # returns [batched_result] unchanged.
        sizes = [arr.shape[0] for arr in inputs]
        split_indices: list[int] = np.cumsum(sizes)[:-1].tolist()
        return list(np.split(result, split_indices))

    def _ort_shape_to_v2(self, shape: list) -> list[int]:
        """Convert an ORT shape (with str entries for dynamic dims) to a V2 shape (-1 for dynamic)."""
        # ORT reports symbolic dims as strings (e.g. 'unk__4862'); V2 uses -1.
        return [-1 if isinstance(s, str) else int(s) for s in shape]

    def _configure_providers(self) -> list[str | tuple]:
        """Helper method for creating providers array for InferenceSession. Outputs list"""
        providers: list = self._config.options["providers"] or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        if "TensorrtExecutionProvider" in self._config.options["providers"]:
            import tensorrt  # ruff: ignore[F401]

            trt_provider_options: dict[str, Any] = {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "/tmp/trt_cache",
                "trt_timing_cache_enable": True,
                "trt_context_memory_sharing_enable": True,
                # Profile shapes match config.pbtxt trt_profile_*_shapes.
                "trt_profile_min_shapes": "Input-Branch1:1x10x10x60x16",
                "trt_profile_max_shapes": "Input-Branch1:250x10x10x60x16",
                "trt_profile_opt_shapes": "Input-Branch1:100x10x10x60x16",
            }

            if self._config.options["trt_provider_options"]:
                trt_provider_options = trt_provider_options | self._config.options.get(
                    "trt_provider_options", {}
                )

            idx = providers.index("TensorrtExecutionProvider")
            providers[idx] = ("TensorrtExecutionProvider", trt_provider_options)

        return providers
