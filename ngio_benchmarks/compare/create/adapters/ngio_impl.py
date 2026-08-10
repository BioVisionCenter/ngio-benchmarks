"""ngio: create the container, set level 0, consolidate the pyramid.

`consolidate` has three modes with very different memory behaviour -- that is
what the `internal` `consolidate` block exists to compare. Here it runs in its
default mode, because the question is what ngio does when you do not tell it
anything, which is what every other library in this table is also being asked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.measure import Measured

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

#: ngio's default consolidation is dask-backed and chunk-bounded.
_DOWNSAMPLE = "dask/mean"


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """Write the whole pyramid, container included."""
    from ngio import create_empty_ome_zarr

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
            ngff_version=spec.ngff_version,
            overwrite=True,
        )
        image = container.get_image(path="0")
        image.set_array(patch=data)
        image.consolidate()

    return Measured(create, extra={"target": str(path), "downsample": _DOWNSAMPLE})
