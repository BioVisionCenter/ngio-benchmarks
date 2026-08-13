"""dask.array over the same zarr store: chunk-parallel access.

The interesting column, because dask is the one peer here that is not trying to
be fast at a single read -- it is trying to overlap many. On a full read that
usually wins; on a small region it usually loses to the graph it had to build
to get there, and the aligned/straddling pair shows whether the scheduler
recovers any of the amplification.

`.compute()` is inside the timed callable on purpose. A dask column that
measured only graph construction would be the fastest in the table and would
mean nothing.

The write goes through `da.to_zarr`, not `da.store(..., lock=False)`. That is
not a style choice -- `reports/ngio-upstream-write-path.md` measured the
latter racing and losing updates on a sharded target, because a patch chunked
to `spec.chunks` covers only a fraction of the shard zarr has to
read-modify-write. Since dask 2025.11, `to_zarr` onto an existing array asks
the target for its write unit, rechunks the patch to a multiple of it, and
only then stores -- correctly, with no lock, on a full write or a straddling
region alike. That makes this row a floor worth reading ngio against: what
writing this correctly costs with no ngio layer on top.
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
#: The 2025.11 floor is load-bearing: `to_zarr`'s write-unit rechunk onto an
#: existing array is what lets the write below skip a lock and stay correct.
REQUIRES = ("dask[array]>=2025.11", "zarr>=3.1.6", "numpy>=2")
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
        # `region=index` covers `write_full` the same as a partial write:
        # `region` selecting every element rechunks and stores exactly like a
        # plain `to_zarr` would, so there is no separate full-write branch.
        da.to_zarr(patch, destination, region=index)

    return Measured(write, extra={"target": str(path)})
