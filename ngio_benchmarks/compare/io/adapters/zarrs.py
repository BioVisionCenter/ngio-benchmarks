"""zarr-python driven by the `zarrs` (Rust) codec pipeline.

Not a second API: `zarrs` installs a replacement for zarr-python's codec
pipeline, so the measured code is byte-for-byte the `zarr` adapter's and the
only difference is who decodes the chunks. That is exactly what makes the pair
worth having -- the delta between these two columns is the codec pipeline and
nothing else.

An implementation here and an axis in `compare-create`, where five different
libraries sit above the pipeline rather than one module. `compare._pipeline`
holds the pin, the format restriction and the reasoning that both share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare import _pipeline
from ngio_benchmarks.compare.io.adapters import zarr_python

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec
    from ngio_benchmarks.core.measure import Measured

NAME = "zarrs"
DISTRIBUTION = "zarrs"
REQUIRES = ("zarr>=3.1.6", *_pipeline.REQUIRES[NAME], "numpy>=2")
SUPPORTS = zarr_python.SUPPORTS
#: zarrs implements the zarr v3 codec pipeline. There is no v2 equivalent to
#: replace, so a v2 image here would silently measure plain zarr-python twice.
FORMATS = _pipeline.FORMATS[NAME]
#: The codec pipeline is Rust, so the decoded chunks are allocated
#: outside Python and `tracemalloc` under-reports this column.
NATIVE = NAME in _pipeline.NATIVE
PYTHON = None


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """Install the zarrs pipeline, then defer to the zarr adapter."""
    _pipeline.install(NAME)
    return zarr_python.build(op, spec, root, impl=NAME)
