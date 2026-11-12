from . import inference_backend, server

__all__ = ["inference_backend", "server"]
# NOTE: `__version__` is not defined because this package is built using 'setuptools-scm' --
#   use `importlib.metadata.version(...)` if you need to access version info at runtime.
