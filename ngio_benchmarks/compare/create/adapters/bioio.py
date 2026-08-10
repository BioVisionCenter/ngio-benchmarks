"""bioio-ome-zarr: explicit per-level shapes rather than scale factors.

The odd one out in this table, and interestingly so. Every other writer is told
how much to downsample and works out the level shapes; `OMEZarrWriter` is told
the shapes and works out the downsampling. That makes it the only one where a
non-power-of-two pyramid is a first-class thing to ask for, and it means the
level count here cannot silently differ from what was requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.measure import Measured

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


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """Write the pyramid with `OMEZarrWriter.write_full_volume`."""
    from bioio_ome_zarr.writers import OMEZarrWriter

    data = _ops.source(spec, root)
    path = _ops.target(root, NAME, spec)
    y, x = spec.pixelsize
    pixel_size = [1.0] * (len(spec.shape) - 2) + [y, x]

    def create() -> None:
        writer = OMEZarrWriter(
            store=str(path),
            # The one writer told the pyramid directly rather than through
            # factors, so `spec.level_shapes()` goes in unchanged. It is also
            # the reference the audit checks every other writer against.
            level_shapes=[list(s) for s in spec.level_shapes()],
            dtype=spec.dtype,
            chunk_shape=tuple(spec.chunks),
            shard_shape=tuple(spec.shards) if spec.shards else None,
            zarr_format=spec.zarr_format,
            axes_names=list(spec.axes),
            physical_pixel_size=pixel_size,
        )
        writer.write_full_volume(data)

    return Measured(
        create, extra={"target": str(path), "downsample": "resize (default)"}
    )
