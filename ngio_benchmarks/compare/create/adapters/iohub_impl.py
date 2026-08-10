"""iohub: an HCS-shaped writer, measured on a single position.

Written for high-content screening, so its natural unit is a plate of positions
rather than one image. Measured here on a single FOV, which is the only shape
comparable with the rest of the table -- but worth knowing when reading the
number, because some of what it costs is machinery for a case this suite is not
exercising.

Requires Python 3.12, while ngio supports 3.11. Its environment therefore pins
its own interpreter rather than inheriting the parent's, which is the whole
argument for one environment per implementation stated as a single line of
configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "iohub"
DISTRIBUTION = "iohub"
REQUIRES = ("iohub>=0.3.10", "numpy>=2")
SUPPORTS = frozenset({"create_pyramid"})
#: `open_ome_zarr` writes NGFF 0.4 or 0.5.
FORMATS = frozenset({2, 3})
PYTHON = "3.12"
REPEATS = 3


def _tczyx(spec: ImageSpec, data):
    """Reshape to the (T, C, Z, Y, X) an iohub position always holds.

    iohub's FOV layout is 5D by definition, so a 4D spec is given a leading
    singleton time axis rather than being reported unsupported. The pixels and
    the pyramid are unchanged, so the column stays comparable -- but the note
    says so, because a reshape the other writers did not need is a difference
    this suite introduced.
    """
    missing = 5 - len(spec.shape)
    if missing <= 0:
        return data, ""
    return (
        data.reshape((1,) * missing + tuple(spec.shape)),
        f"input reshaped to 5D ({missing} leading axis added)",
    )


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """Create a position, write level 0, compute the pyramid."""
    from iohub import open_ome_zarr

    data, note = _tczyx(spec, _ops.source(spec, root))
    path = _ops.target(root, NAME, spec)
    downsample = set(spec.downsample_axes)
    channels = (
        [f"Channel {i + 1}" for i in range(spec.shape[0])]
        if "c" in spec.axes
        else ["Channel 1"]
    )
    chunks = (1,) * (5 - len(spec.chunks)) + tuple(spec.chunks)
    shards_ratio = None
    if spec.shards:
        ratio = tuple(
            shard // chunk
            for shard, chunk in zip(spec.shards, spec.chunks, strict=True)
        )
        shards_ratio = (1,) * (5 - len(ratio)) + ratio

    def create() -> None:
        with open_ome_zarr(
            str(path),
            layout="fov",
            mode="w",
            channel_names=channels,
            version="0.4" if spec.zarr_format == 2 else "0.5",
        ) as position:
            position.create_image("0", data, chunks=chunks, shards_ratio=shards_ratio)
            if spec.levels > 1:
                # `dims` pinned: iohub defaults to {"z", "y", "x"}, so without
                # this it halves z as well and builds a pyramid with half the
                # voxels per level of the one ngio, bioio and ome-zarr-py
                # build -- a difference that reads as iohub being fast.
                position.initialize_pyramid(levels=spec.levels, dims=downsample)
                position.compute_pyramid(dims=downsample)

    return Measured(
        create,
        note,
        extra={"target": str(path), "downsample": "iohub pyramid (default)"},
    )
