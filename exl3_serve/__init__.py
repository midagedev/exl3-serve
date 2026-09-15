"""exl3-serve: a llama-server-compatible HTTP front for ExLlamaV3."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("exl3-serve")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+source"
