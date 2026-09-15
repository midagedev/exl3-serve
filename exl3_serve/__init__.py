"""exl3-serve: a llama-server-compatible HTTP front for ExLlamaV3."""
import os

# Must be set before exllamav3 (and therefore torch) is imported anywhere in
# this process, so cuda:<n> device names match nvidia-smi's indices -- the
# /props placement reports GPU<n> with that numbering.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from importlib.metadata import PackageNotFoundError, version  # noqa: E402

try:
    __version__ = version("exl3-serve")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+source"
