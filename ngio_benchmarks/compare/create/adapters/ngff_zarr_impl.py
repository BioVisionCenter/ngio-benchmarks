"""ngff-zarr: the reference implementation of the NGFF data model itself.

Three steps rather than one call -- describe the image, build the multiscales,
write them -- and all three are inside the timing, because together they are
what the other libraries do in a single call.

Its default downsampling goes through `itkwasm`, a WebAssembly build of ITK.
That is a genuinely different trade from everyone else here: better anti-aliasing,
a much heavier dependency, and a runtime cost that shows up in this column. The
`DASK_IMAGE_NEAREST` alternative would be closer to what the others do, but
choosing it for them would hide the choice this library actually makes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "ngff-zarr"
DISTRIBUTION = "ngff-zarr"
REQUIRES = ("ngff-zarr>=0.41", "numpy>=2")
SUPPORTS = frozenset({"create_pyramid"})
#: `to_ngff_zarr` writes NGFF 0.4 or 0.5, which is zarr v2 or v3 respectively.
FORMATS = frozenset({2, 3})
PYTHON = None
REPEATS = 3


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """Describe, downsample, write."""
    import ngff_zarr

    data = _ops.source(spec, root)
    path = _ops.target(root, NAME, spec)
    dims = list(spec.axes)
    y, x = spec.pixelsize
    # Every dim, not just the two with a real pixel size: ngff-zarr indexes the
    # scale mapping by dim name while building the coordinate transformations,
    # so a missing key is a KeyError rather than a default.
    scale = {name: {"y": y, "x": x}.get(name, 1.0) for name in dims}
    chunks = dict(zip(dims, spec.chunks, strict=True))
    shards = None
    if spec.shards:
        shards = {
            name: shard // chunk
            for name, shard, chunk in zip(dims, spec.shards, spec.chunks, strict=True)
        }

    def create() -> None:
        image = ngff_zarr.to_ngff_image(data, dims=dims, scale=scale)
        multiscales = ngff_zarr.to_multiscales(
            image,
            # `explicit_ones`: ngff-zarr indexes every spatial dim of the
            # factor mapping, so an axis left out is a KeyError rather than an
            # axis left alone.
            scale_factors=_ops.scale_factors(spec, explicit_ones=True),
            chunks=chunks,
        )
        ngff_zarr.to_ngff_zarr(
            str(path),
            multiscales,
            version="0.4" if spec.zarr_format == 2 else "0.5",
            overwrite=True,
            chunks_per_shard=shards,
        )

    return Measured(
        create,
        extra={"target": str(path), "downsample": "itkwasm-gaussian (default)"},
    )
