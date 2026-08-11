"""bioio-ome-zarr: explicit per-level shapes rather than scale factors.

The odd one out in this table, and interestingly so. Every other writer is told
how much to downsample and works out the level shapes; `OMEZarrWriter` is told
the shapes and works out the downsampling. That makes it the only one where a
non-power-of-two pyramid is a first-class thing to ask for, and it means the
level count here cannot silently differ from what was requested.

It is also the only writer with no filter argument at all. The downsampling is
`resize(..., order=0)` -- nearest neighbour, no anti-aliasing -- hard-coded in
all three of its write paths. So `METHODS` has exactly two entries, and asking
this column for a mean or a gaussian gets `unsupported` rather than a number
produced by a filter nobody asked for. The previous version of this adapter
recorded `"resize (default)"` for it, which read as an anti-aliased resize and
was wrong.

What it does offer instead is three ways to feed it: the whole volume at once,
a batch of timepoints at a time, or region by region. Those have very different
memory profiles and are swept through `[options.bioio] writer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.measure import Measured, Unsupported

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "bioio"
DISTRIBUTION = "bioio-ome-zarr"
REQUIRES = ("bioio-ome-zarr>=3.6", "numpy>=2")
SUPPORTS = frozenset({"create_pyramid"})
FORMATS = frozenset({2, 3})
PYTHON = None
REPEATS = 3

#: Only what it actually does. `resize(order=0)` is a decimation, so `nearest`
#: is the one canonical name that is honestly the same operation.
METHODS = {
    "default": "resize order=0",
    "nearest": "resize order=0",
}

#: No `method_native`: there is no filter argument on `OMEZarrWriter` to name.
METHOD_NATIVE = False

OPTIONS = {
    #: `full_volume` hands it everything; `timepoints` walks the t axis in
    #: batches, which bounds the memory a level costs. Closed -- these are
    #: methods on the writer, not values it validates.
    "writer": {"full_volume": "write_full_volume", "timepoints": "write_timepoints"},
}


def build(
    op: str,
    spec: ImageSpec,
    root: Path,
    *,
    method: str = "default",
    method_native: str | None = None,
    writer: str = "write_full_volume",
    **options: str,
) -> Measured:
    """Write the pyramid with `OMEZarrWriter`."""
    from bioio_ome_zarr.writers import OMEZarrWriter

    if method_native is not None:
        raise Unsupported(
            "bioio-ome-zarr takes no downsampling filter; it is fixed at "
            "resize(order=0)"
        )

    data = _ops.source(spec, root)
    path = _ops.target(root, NAME, spec)
    y, x = spec.pixelsize
    pixel_size = [1.0] * (len(spec.shape) - 2) + [y, x]
    compressor = _compressor(spec)

    def make() -> OMEZarrWriter:
        return OMEZarrWriter(
            store=str(path),
            # The one writer told the pyramid directly rather than through
            # factors, so `spec.level_shapes()` goes in unchanged. It is also
            # the reference the audit checks every other writer against.
            level_shapes=[list(s) for s in spec.level_shapes()],
            dtype=spec.dtype,
            chunk_shape=tuple(spec.chunks),
            shard_shape=tuple(spec.shards) if spec.shards else None,
            compressor=compressor,
            zarr_format=spec.zarr_format,
            axes_names=list(spec.axes),
            physical_pixel_size=pixel_size,
        )

    if writer == "write_timepoints" and "t" not in spec.axes:
        # `write_timepoints` raises on an image with no t axis. Reported as a
        # property of the pairing rather than as a crash: this writer is not
        # broken, it is being asked to batch along an axis the image does not
        # have, and the fix is a 5D image spec.
        raise Unsupported(
            f"bioio's write_timepoints needs a t axis; {spec.name} has "
            f"{', '.join(spec.axes)}"
        )

    def create() -> None:
        getattr(make(), writer)(data)

    return Measured(
        create,
        extra={"target": str(path), "method_native": "resize order=0"},
        setup=_ops.fresh(path),
    )


def _compressor(spec: ImageSpec):
    """Translate `spec.compressors` into the codec object bioio wants.

    Format-dependent by construction: `OMEZarrWriter` passes it straight to
    `create_array`, so zarr v2 needs a `numcodecs` codec and v3 a `zarr.codecs`
    one. `None` means "use bioio's default", which is Blosc/zstd/3 -- and is
    what produced a store a third smaller than everyone else's back when this
    argument was not passed at all.
    """
    if spec.compressors == "auto":
        return None
    from ngio_benchmarks.core.images import zarr_compressors

    return zarr_compressors(spec)
