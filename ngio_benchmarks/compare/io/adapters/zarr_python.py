"""zarr-python used directly: the floor every other row is measured against.

No OME layer, no convenience, no index translation -- `arr[sel]` and
`arr[sel] = data`. Every other implementation in this suite is doing at least
this much work, so a column that is slower than `zarr` is spending the
difference on something, and a column that is faster is not going through
zarr-python at all.

The same module backs the `zarrs` adapter, which is this code under a different
codec pipeline. See `zarrs.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.io import _ops
from ngio_benchmarks.compare.io._fixture import store_path
from ngio_benchmarks.core.images import zarr_compressors
from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "zarr"
DISTRIBUTION = "zarr"
REQUIRES = ("zarr>=3.1.6", "numpy>=2")
SUPPORTS = frozenset(_ops.READS + _ops.WRITES)
FORMATS = frozenset({2, 3})
PYTHON = None


def open_level(spec: ImageSpec, root: Path):
    """Open level 0 of the shared read fixture."""
    import zarr

    return zarr.open_array(store=store_path(root, spec) / "0", mode="r")


def create_level(path: Path, spec: ImageSpec):
    """Create an empty array matching `spec`, ready to be written into."""
    import zarr

    return zarr.create_array(
        store=path,
        shape=spec.shape,
        dtype=spec.dtype,
        chunks=spec.chunks,
        shards=spec.shards,
        compressors=zarr_compressors(spec),
        zarr_format=spec.zarr_format,
        overwrite=True,
    )


def build(op: str, spec: ImageSpec, root: Path, impl: str = NAME) -> Measured:
    """The measured callable for one operation."""
    index = _ops.region(spec, op)

    if _ops.is_read(op):
        array = open_level(spec, root)
        return Measured(lambda: array[index])

    path = _ops.target(root, impl, op, spec)
    array = create_level(path, spec)
    data = _ops.patch(spec, op, root)

    def write() -> None:
        array[index] = data

    return Measured(write, extra={"target": str(path)})
