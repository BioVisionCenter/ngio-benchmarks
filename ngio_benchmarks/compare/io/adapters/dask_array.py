"""dask.array over the same zarr store: chunk-parallel access.

The interesting column, because dask is the one peer here that is not trying to
be fast at a single read -- it is trying to overlap many. On a full read that
usually wins; on a small region it usually loses to the graph it had to build
to get there, and the aligned/straddling pair shows whether the scheduler
recovers any of the amplification.

`.compute()` is inside the timed callable on purpose. A dask column that
measured only graph construction would be the fastest in the table and would
mean nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.io import _ops
from ngio_benchmarks.compare.io.adapters import zarr_python
from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "dask"
DISTRIBUTION = "dask"
REQUIRES = ("dask[array]>=2024.1", "zarr>=3.1.6", "numpy>=2")
SUPPORTS = frozenset(_ops.READS + _ops.WRITES)
FORMATS = frozenset({2, 3})
PYTHON = None


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """The measured callable for one operation."""
    import dask.array as da

    index = _ops.region(spec, op)

    if _ops.is_read(op):
        source = zarr_python.open_level(spec, root)
        array = da.from_array(source, chunks=spec.chunks)
        return Measured(lambda: array[index].compute())

    path = _ops.target(root, NAME, op, spec)
    destination = zarr_python.create_level(path, spec)
    data = _ops.patch(spec, op, root)
    patch = da.from_array(data, chunks=spec.chunks)

    def write() -> None:
        # `regions` rather than `to_zarr`: a region write is the operation
        # being measured, and `to_zarr` would only ever express the full case.
        #
        # A bare tuple, not a list of one: `da.store` wraps a tuple into a
        # single-element list itself, so wrapping it here would hand the target
        # a tuple nested one level too deep.
        da.store(patch, destination, regions=index, lock=False)

    return Measured(write, extra={"target": str(path)})
