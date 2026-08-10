"""zarr-python driven by the `zarrs` (Rust) codec pipeline.

Not a second API: `zarrs` installs a replacement for zarr-python's codec
pipeline, so the measured code is byte-for-byte the `zarr` adapter's and the
only difference is who decodes the chunks. That is exactly what makes the pair
worth having -- the delta between these two columns is the codec pipeline and
nothing else.

Set once at build time rather than around the timed callable, because
`zarr.config.set` is a context the pipeline is chosen from at array-open time,
and toggling it inside the measurement would time the toggle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.io.adapters import zarr_python

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec
    from ngio_benchmarks.core.measure import Measured

NAME = "zarrs"
DISTRIBUTION = "zarrs"
REQUIRES = ("zarr>=3.1.6", "zarrs>=0.2.3", "numpy>=2")
SUPPORTS = zarr_python.SUPPORTS
#: zarrs implements the zarr v3 codec pipeline. There is no v2 equivalent to
#: replace, so a v2 image here would silently measure plain zarr-python twice.
FORMATS = frozenset({3})
#: The codec pipeline is Rust, so the decoded chunks are allocated
#: outside Python and `tracemalloc` under-reports this column.
NATIVE = True
PYTHON = None


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """Install the zarrs pipeline, then defer to the zarr adapter."""
    import zarr

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    return zarr_python.build(op, spec, root, impl=NAME)
