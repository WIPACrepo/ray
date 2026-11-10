import argparse
import asyncio
import dataclasses as dc
import importlib.metadata
import logging
import os
import signal
import time
from typing import Annotated, Any, ClassVar
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import numpy as np


@dc.dataclass
class BackendConfig(BaseModel):
    runtime: str
    model_repo: str
    model_name: str
    model_version: str
    model_path: str | None = None
    options: dict[str, Any] = dc.field(default_factory=dict)


class InferenceBackend(ABC):
    # Map from V2 datatype strings to numpy dtypes — mirrors the client's _V2_DTYPE_TO_NUMPY.
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
        self.config = config

    def get_module_path(self) -> str:
        return self.config.model_path or os.path.join(
            self.config.model_repo, self.config.model_name, self.config.model_version
        )

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def warm(self) -> None: ...
    def close(self) -> None: ...


_REGISTRY: dict[str, type[InferenceBackend]] = {}


def register_backend(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def create_backend(config: BackendConfig) -> InferenceBackend:
    try:
        return _REGISTRY[config.runtime](config)
    except KeyError:
        raise ValueError(f"Unknown runtime {config.runtime!r}; have {list(_REGISTRY)}")


@register_backend("onnx")
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
            self.get_module_path(), sess_options=sess_options, providers=providers
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

    def _configure_providers(self):
        providers: list = self.config.options["providers"] or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        if "TensorrtExecutionProvider" in self.config.options["providers"]:
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

            if self.config.options["trt_provider_options"]:
                trt_provider_options = (
                    trt_provider_options | self.config.options["trt_provider_options"]
                )

            idx = providers.index("TensorrtExecutionProvider")
            providers[idx] = ("TensorrtExecutionProvider", trt_provider_options)

        return providers
