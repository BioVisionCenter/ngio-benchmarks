"""ngio: create the container, set level 0, consolidate the pyramid.

`consolidate` takes two arguments that both change what this column measures.
`mode` picks the machinery -- `numpy` materialises a whole level, `dask` and
`coarsen` are chunk-bounded -- and `order` picks the filter. They are not
independent: only `coarsen` computes a genuine box mean, because it is the only
one that aggregates rather than interpolating, and its `order` argument is a
compatibility shim rather than a real interpolation order.

That is why `METHODS` maps `mean` onto `coarsen` and everything else onto a
zoom. A config sweeping `mode` directly gets whichever `order` the method
implies, which is the honest way round: the mode is the thing being compared and
the filter is what has to be held equal.

One thing to know before reading this column's `pipeline=zarrs` row: **it is not
one.** ngio wraps every store in `NgioStore`, a `zarr.storage.WrapperStore`
subclass carrying its retry logic, and `zarrs` installs its pipeline only for
the store types it recognises -- `LocalStore`, `ObjectStore`, `FsspecStore`.
Handed anything else, zarr-python keeps `BatchedCodecPipeline` silently, so the
row measures zarr-python. Verified rather than assumed: the array ngio hands
back reports `BatchedCodecPipeline` where a plain `zarr.create_array` in the same
process reports `ZarrsCodecPipeline`, and passing a raw `LocalStore` to
`create_empty_ome_zarr` does not help -- it arrives wrapped either way.

The timing agrees, and is the general tell for this: wall *and* CPU seconds both
unchanged means the same code ran. The audit columns cannot catch it, because
the store written is byte-identical whichever pipeline encoded it.

One asymmetry to know about when sweeping `mode` and `method` together:
`coarsen` **ignores** `order`. `on_disk_zoom` forwards it to the numpy and dask
paths but calls `_on_disk_coarsen(source, target)` without it, so coarsening is
always a mean. `method = "nearest"` with `mode = "coarsen"` therefore produces a
mean, and `method_native` says so rather than repeating back the order it was
handed. The store size agrees: a coarsen `nearest` writes the same bytes as a
coarsen `mean`, where every real decimation in this table writes fewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.images import zarr_compressors
from ngio_benchmarks.core.measure import Measured, Unsupported

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "ngio"
DISTRIBUTION = "ngio"
REQUIRES = ("ngio",)
SUPPORTS = frozenset({"create_pyramid"})
FORMATS = frozenset({2, 3})
PYTHON = None
REPEATS = 3

#: Canonical downsampling name -> `(mode, order)` for `Image.consolidate`.
#:
#: `default` is ngio's own default, spelled out rather than left implicit, so
#: that the row labelled `default` and the row labelled `linear` being identical
#: is visible in the table instead of being a fact about ngio you had to know.
#:
#: No `gaussian`: ngio's zoom takes an interpolation order, not a kernel, and
#: presenting `cubic` as a gaussian would be a translation this suite invented.
METHODS = {
    "default": ("dask", "linear"),
    "mean": ("coarsen", "linear"),
    "nearest": ("numpy", "nearest"),
    "linear": ("dask", "linear"),
}

#: No `method_native`: ngio does not take a filter by name, so `mode` and
#: `order` below are the pass-through and there is nothing else to offer.
METHOD_NATIVE = False

#: `mode` overrides the machinery `METHODS` implied, leaving its filter alone.
#: Open rather than closed: ngio validates the value itself, and a closed axis
#: here would go stale the moment ngio grows a fourth mode.
OPTIONS = {
    "mode": ["dask", "numpy", "coarsen"],
    "order": ["linear", "nearest", "cubic"],
}


def build(
    op: str,
    spec: ImageSpec,
    root: Path,
    *,
    method: str = "default",
    method_native: str | None = None,
    **options: str,
) -> Measured:
    """Write the whole pyramid, container included."""
    from typing import Any, cast

    from ngio import create_empty_ome_zarr

    if method_native is not None:
        raise Unsupported(
            "ngio names its downsampling by mode and order rather than by "
            "filter; sweep [options.ngio] mode / order instead"
        )
    mode, order = METHODS[method]
    mode = options.get("mode", mode)
    order = options.get("order", order)
    # `_on_disk_coarsen` is called without the order, so coarsen always
    # aggregates with `np.mean` whatever it was asked for. Recorded as what it
    # will do rather than as what it was told: a row reading `coarsen/nearest`
    # would be a mean labelled as a decimation, and the store size gives it
    # away anyway -- a coarsen `nearest` writes the same bytes as a mean.
    native = f"{mode}/mean (order ignored)" if mode == "coarsen" else f"{mode}/{order}"

    data = _ops.source(spec, root)
    path = _ops.target(root, NAME, spec)

    # One factor per axis, applied at every level. Pinned rather than left at
    # "auto" so the pyramid is the one the spec asked for and not the one this
    # ngio happens to default to -- the comparison is of writers, and a column
    # that quietly built a different pyramid is not in it.
    halved = set(spec.downsample_axes)
    scaling = tuple(2.0 if name in halved else 1.0 for name in spec.axes)

    def create() -> None:
        container = create_empty_ome_zarr(
            store=path,
            shape=spec.shape,
            axes_names=list(spec.axes),
            channels_meta=[f"Channel {i + 1}" for i in range(spec.shape[0])]
            if spec.axes[0] == "c"
            else None,
            levels=spec.levels,
            scaling_factors=scaling,
            pixelsize=spec.pixelsize,
            chunks=spec.chunks,
            shards=spec.shards,
            dtype=spec.dtype,
            # Passed rather than left at ngio's default, which is what every
            # other writer here is also now told. Without it each library used
            # its own codec and the `bytes` column compared codecs.
            compressors=zarr_compressors(spec),
            ngff_version=spec.ngff_version,
            overwrite=True,
        )
        image = container.get_image(path="0")
        image.set_array(patch=data)
        image.consolidate(order=cast("Any", order), mode=cast("Any", mode))

    return Measured(
        create,
        extra={"target": str(path), "method_native": native},
        setup=_ops.fresh(path),
    )
