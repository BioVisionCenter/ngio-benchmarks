"""ngio, through its public image API.

Reads go through `open_image` and the slicing keywords rather than the `Roi`
objects the internal `roi` block uses. Two reasons: `ngio.common.Roi` is not
part of the public surface an adapter should depend on, and a keyword slice is
the same *kind* of request every other library in this suite receives, so the
comparison is of the machinery underneath rather than of two different ways of
asking.

The container ngio writes into is created during setup and excluded from the
timing, which is the right split -- the peers here are also handed an existing
array to write into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.io import _ops
from ngio_benchmarks.compare.io._fixture import store_path
from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "ngio"
DISTRIBUTION = "ngio"
REQUIRES = ("ngio",)
SUPPORTS = frozenset(_ops.READS + _ops.WRITES)
FORMATS = frozenset({2, 3})
PYTHON = None


def _slicing(spec: ImageSpec, op: str) -> dict[str, slice]:
    """The trailing-axis slices as ngio's keyword arguments.

    Empty for a full read: ngio takes the whole array when nothing is named,
    and passing explicit full slices would measure a slicing path the other
    libraries are not being asked to take.
    """
    if _ops.is_full(op):
        return {}
    index = _ops.region(spec, op)
    return {
        name: index[i] for i, name in enumerate(spec.axes) if i >= len(spec.shape) - 2
    }


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """The measured callable for one operation."""
    from ngio import open_image

    slicing = _slicing(spec, op)

    if _ops.is_read(op):
        image = open_image(store_path(root, spec), path="0", mode="r")
        return Measured(lambda: image.get_as_numpy(**slicing))

    path = _ops.target(root, NAME, op, spec)
    image = _create(path, spec)
    data = _ops.patch(spec, op, root)
    return Measured(
        lambda: image.set_array(patch=data, **slicing), extra={"target": str(path)}
    )


def _create(path: Path, spec: ImageSpec):
    """An empty single-level container for the write operations."""
    from ngio import create_empty_ome_zarr

    container = create_empty_ome_zarr(
        store=path,
        shape=spec.shape,
        axes_names=list(spec.axes),
        channels_meta=[f"Channel {i + 1}" for i in range(spec.shape[0])]
        if spec.axes[0] == "c"
        else None,
        levels=1,
        pixelsize=spec.pixelsize,
        chunks=spec.chunks,
        shards=spec.shards,
        ngff_version=spec.ngff_version,
        overwrite=True,
    )
    return container.get_image(path="0")
