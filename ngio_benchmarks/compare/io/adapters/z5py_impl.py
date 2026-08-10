"""z5py: a C++ zarr/n5 implementation with a h5py-shaped API.

Included because it is the oldest of these and comes at the problem from the
HDF5 side rather than the zarr side, so it is the one column whose performance
characteristics were not designed against the same benchmarks as everyone
else's.

Two practical limits, both declared rather than discovered: it publishes no
manylinux aarch64 wheel, so on Linux ARM this environment fails to install and
the column reports `unavailable`; and its compression vocabulary is its own, so
`auto` here means whatever z5py defaults to rather than what zarr-python would
have chosen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.io import _ops
from ngio_benchmarks.compare.io._fixture import store_path
from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "z5py"
DISTRIBUTION = "z5py"
REQUIRES = ("z5py>=3.0.2", "numpy>=2")
#: Sharding is a zarr v3 feature z5py does not implement; a sharded spec is
#: excluded by `build` rather than silently written unsharded.
SUPPORTS = frozenset(_ops.READS + _ops.WRITES)
FORMATS = frozenset({2, 3})
#: A C++ implementation: its buffers never pass through the Python
#: allocator, so `tracemalloc` cannot account for them.
NATIVE = True
PYTHON = None

#: `compressors` label -> z5py's `compression` argument.
_COMPRESSION = {
    "auto": "blosc",
    "none": "raw",
    "zstd": "zstd",
    "blosc": "blosc",
    "lz4": "blosc",
}


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """The measured callable for one operation."""
    import z5py

    from ngio_benchmarks.core.measure import Unsupported

    if spec.shards is not None:
        raise Unsupported("z5py does not implement zarr v3 sharding")

    index = _ops.region(spec, op)

    if _ops.is_read(op):
        handle = z5py.File(
            str(store_path(root, spec)), "r", zarr_format=spec.zarr_format
        )
        dataset = handle["0"]
        return Measured(lambda: dataset[index])

    path = _ops.target(root, NAME, op, spec)
    handle = z5py.File(str(path), "w", zarr_format=spec.zarr_format)
    dataset = handle.create_dataset(
        "0",
        shape=spec.shape,
        dtype=spec.dtype,
        chunks=spec.chunks,
        compression=_COMPRESSION[spec.compressors],
    )
    data = _ops.patch(spec, op, root)

    def write() -> None:
        dataset[index] = data

    return Measured(write, extra={"target": str(path)})
