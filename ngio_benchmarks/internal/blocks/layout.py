"""Same bytes, different on-disk layout: what does the layout cost to read?

`get`/`set` are deliberately absent as blocks of their own -- they are a thin
layer over zarr, adding a roughly constant ~0.8 ms, so timing them mostly
re-measures zarr. What is worth measuring is the layout underneath them. The
`compare-io` suite asks the complementary question: what does that thin layer
cost against the libraries that do not have it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.core.measure import Measured
from ngio_benchmarks.internal.fixtures import image_fixture

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

AXES = {
    # A *closed* axis: each value is a bundle of fixture kwargs, so the labels
    # are the vocabulary. A config can subset these but cannot invent one,
    # which is right -- chunk and shard shapes are not independent knobs
    # (`sharded` + `uncompressed` is not a case anyone wants), and a shard
    # shape is not something to spell out per experiment.
    "layout": {
        "chunks512": {"chunks": (1, 1, 512, 512)},
        "chunks128": {"chunks": (1, 1, 128, 128)},
        "sharded": {"chunks": (1, 1, 128, 128), "shards": (1, 4, 1024, 1024)},
        "uncompressed": {"chunks": (1, 1, 512, 512), "compressors": None},
    },
    # Open, so `[axes.layout] z = [16, 64, 128]` crosses depth with layout.
    "z": [16],
}

REPEATS = 3


def run(root: Path, *, layout: dict[str, Any], z: int) -> Measured:
    """Read a whole level back as numpy."""
    from ngio import open_image

    path = image_fixture(root, (2, z, 1024, 1024), **layout)
    return Measured(open_image(path, path="0", mode="r").get_as_numpy)
