"""Pyramid consolidation: which mode should I use, and will it fit?

`numpy` materialises a whole level (plus the zoom output twice, via
`_stacked_zoom`), so its peak tracks the data and it has a hard ceiling. `dask`
and `coarsen` are chunk-bounded and stay roughly flat as the data grows. That
difference is invisible in a timing -- `numpy` is also the fastest -- which is
why peak memory is a first-class column here and not an extra.
"""

from __future__ import annotations

import math
import shutil
from typing import TYPE_CHECKING, Any, cast

from ngio_benchmarks.core.data import synthetic
from ngio_benchmarks.core.measure import MB, Measured

if TYPE_CHECKING:
    from pathlib import Path

AXES = {
    "mode": ["dask", "numpy", "coarsen"],
    # Open, so `[axes.consolidate] z = [1024]` sweeps past what is declared
    # here. z * 512 * 512 * 2 bytes: 16 -> 8 MB, 64 -> 32 MB, 256 -> 128 MB.
    "z": [16, 64, 256],
    # Open, both default to 1 so an existing config that never names them
    # measures exactly what it measured before these axes existed. `t` above
    # 1 adds a leading axis rather than growing an existing one: a time series
    # is a different shape from a bigger volume, not the same one scaled up.
    "c": [1],
    "t": [1],
    # Open, defaults to dask's own `array.chunk-size`. Any other value is set
    # for the duration of the consolidation and nothing else.
    #
    # This is the knob that decides how large a block `da.to_zarr` groups the
    # write into, and therefore how much of a level has to be resident at once.
    # ngio's `store_dask` takes `max(array.chunk-size, one write unit)`, so on
    # an unsharded image -- where the unit is a 128 KiB chunk -- lowering this
    # is exactly equivalent to capping ngio's budget, without patching ngio.
    # That is the point: the proposed fix can be measured before it is written.
    "chunk_size": ["default"],
}

REPEATS = 3


def run(
    root: Path,
    *,
    mode: str,
    z: int,
    c: int = 1,
    t: int = 1,
    chunk_size: str = "default",
) -> Measured:
    """Build a fresh pyramid; measure only its consolidation."""
    from ngio import create_empty_ome_zarr

    if t > 1:
        shape = (t, c, z, 512, 512)
        axes_names = ["t", "c", "z", "y", "x"]
        chunks = (1, 1, 1, 256, 256)
    else:
        shape = (c, z, 512, 512)
        axes_names = ["c", "z", "y", "x"]
        chunks = (1, 1, 256, 256)
    target = root / f"consolidate_{mode}_{z}_{c}c_{t}t.zarr"
    if target.exists():
        shutil.rmtree(target)
    container = create_empty_ome_zarr(
        store=target,
        shape=shape,
        axes_names=axes_names,
        channels_meta=[f"Channel {i + 1}" for i in range(c)],
        levels=3,
        pixelsize=(0.65, 0.65),
        chunks=chunks,
        overwrite=True,
    )
    image = container.get_image(path="0")
    # Realistic data, not `np.ones`: consolidation writes every level, and
    # uniform data compresses ~2000:1, which would make the write half of the
    # work almost free and flatter all three modes equally but unrealistically.
    image.set_array(patch=synthetic(shape))
    # `mode` is an *open* axis, so a config can reach this with a plain str.
    # Validating it here would only duplicate the check ngio already makes,
    # and worse, would go stale when ngio grows a fourth mode.
    def consolidate() -> None:
        if chunk_size == "default":
            image.consolidate(mode=cast("Any", mode))
            return
        import dask

        # Inside the measured callable, not around the fixture: `to_zarr` reads
        # this when it *builds* the graph, which happens within `consolidate`.
        with dask.config.set({"array.chunk-size": chunk_size}):
            image.consolidate(mode=cast("Any", mode))

    return Measured(consolidate, f"{math.prod(shape) * 2 / MB:.0f} MB")
