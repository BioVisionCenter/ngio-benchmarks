"""tensorstore: the same zarr bytes, decoded outside Python.

Less a peer than a ceiling. tensorstore does the chunk maths and the codec work
in C++ with its own thread pool, so the gap between it and `zarr` is roughly
what zarr-python spends being Python. Read that gap as a budget, not as a
verdict -- none of the OME-Zarr libraries here could adopt it without giving up
the ecosystem they sit in.

One caveat specific to this column: `peak_mb` comes from `tracemalloc`, which
sees Python allocations only. tensorstore's buffers are largely invisible to
it, so its memory numbers are not comparable with the rest of the table. Its
timings are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ngio_benchmarks.compare.io import _ops
from ngio_benchmarks.compare.io._fixture import store_path
from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "tensorstore"
DISTRIBUTION = "tensorstore"
REQUIRES = ("tensorstore>=0.1.85", "numpy>=2")
SUPPORTS = frozenset(_ops.READS + _ops.WRITES)
FORMATS = frozenset({2, 3})
#: Allocates its buffers in C++, so `tracemalloc` reports 0.0 for it --
#: measured, a 128 MiB read shows 0.0 MB traced against 132.8 MB resident.
#: Read `proc_peak_mb` for this column instead.
NATIVE = True
PYTHON = None

#: numpy dtype -> the string zarr v2 metadata wants.
_V2_DTYPE = {
    "uint8": "|u1",
    "int8": "|i1",
    "uint16": "<u2",
    "int16": "<i2",
    "uint32": "<u4",
    "float32": "<f4",
}


def _inner_codecs(spec: ImageSpec) -> list[dict[str, Any]]:
    """The zarr v3 codec chain, below any sharding."""
    codecs: list[dict[str, Any]] = [
        {"name": "bytes", "configuration": {"endian": "little"}}
    ]
    if spec.compressors == "none":
        return codecs
    if spec.compressors == "zstd":
        return [*codecs, {"name": "zstd", "configuration": {"level": 1}}]
    cname = "lz4" if spec.compressors == "lz4" else "zstd"
    return [
        *codecs,
        {
            "name": "blosc",
            "configuration": {"cname": cname, "clevel": 5, "shuffle": "shuffle"},
        },
    ]


def _v3_metadata(spec: ImageSpec) -> dict[str, Any]:
    """Zarr v3 metadata, sharded or not.

    Under sharding the chunk grid is the *shard* and the chunk shape moves
    inside the `sharding_indexed` codec. Getting that inversion wrong writes a
    valid store with the wrong layout, which would show up as a suspiciously
    fast column rather than as an error.
    """
    grid = spec.shards or spec.chunks
    codecs: list[dict[str, Any]] = _inner_codecs(spec)
    if spec.shards:
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": list(spec.chunks),
                    "codecs": codecs,
                },
            }
        ]
    metadata: dict[str, Any] = {
        "shape": list(spec.shape),
        "data_type": spec.dtype,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(grid)},
        },
    }
    if spec.compressors != "auto":
        metadata["codecs"] = codecs
    return metadata


def _v2_metadata(spec: ImageSpec) -> dict[str, Any]:
    """Zarr v2 metadata."""
    metadata: dict[str, Any] = {
        "shape": list(spec.shape),
        "chunks": list(spec.chunks),
        "dtype": _V2_DTYPE[spec.dtype],
    }
    if spec.compressors == "none":
        metadata["compressor"] = None
    elif spec.compressors in ("blosc", "lz4"):
        metadata["compressor"] = {
            "id": "blosc",
            "cname": "lz4" if spec.compressors == "lz4" else "zstd",
            "clevel": 5,
            "shuffle": 1,
        }
    elif spec.compressors == "zstd":
        metadata["compressor"] = {"id": "zstd", "level": 1}
    return metadata


def _spec(path: Path, spec: ImageSpec, *, create: bool) -> dict[str, Any]:
    """The tensorstore spec for opening or creating `path`."""
    driver = "zarr3" if spec.zarr_format == 3 else "zarr"
    request: dict[str, Any] = {
        "driver": driver,
        "kvstore": {"driver": "file", "path": str(path)},
    }
    if not create:
        request["open"] = True
        return request
    request.update(
        {
            "create": True,
            "delete_existing": True,
            "metadata": _v3_metadata(spec)
            if spec.zarr_format == 3
            else _v2_metadata(spec),
        }
    )
    return request


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """The measured callable for one operation."""
    import tensorstore as ts

    index = _ops.region(spec, op)

    if _ops.is_read(op):
        store = ts.open(
            _spec(store_path(root, spec) / "0", spec, create=False)
        ).result()
        # `.result()` inside the callable: tensorstore reads are futures, and
        # timing the future's creation would time nothing at all.
        return Measured(lambda: store[index].read().result())

    path = _ops.target(root, NAME, op, spec)
    store = ts.open(_spec(path, spec, create=True)).result()
    data = _ops.patch(spec, op, root)

    def write() -> None:
        store[index].write(data).result()

    return Measured(write, extra={"target": str(path)})
