"""On-disk fixtures for the internal blocks.

Everything here goes through ngio's **public API only**. That is what lets a
block run unmodified inside an environment holding a different ngio: a private
module path that moved would otherwise break the import and take the fixture
down with it.

The pixels come from `core.data`, shared with the comparison half, so a number
from `internal` and a number from `compare-io` were measured on the same bytes.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

from ngio_benchmarks.core.data import synthetic

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec


def image_fixture(
    root: Path,
    shape: tuple[int, ...],
    *,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None = None,
    compressors: Any = "auto",
    levels: int = 3,
    ngff_version: str = "0.5",
) -> Path:
    """Create (or reuse) an image fixture and return its path.

    The name is derived from the spec, not passed in, so two blocks asking for
    the same image share one store under `keep` instead of each building its
    own. It also means a changed spec lands on a new path rather than silently
    reusing a stale store.
    """
    from ngio import create_empty_ome_zarr

    spec = (shape, chunks, shards, repr(compressors), levels, ngff_version)
    digest = hashlib.blake2s(repr(spec).encode(), digest_size=4).hexdigest()
    path = root / f"img_{'x'.join(str(s) for s in shape)}_{digest}.zarr"
    if path.exists():
        return path
    container = create_empty_ome_zarr(
        store=path,
        shape=shape,
        axes_names=["c", "z", "y", "x"],
        channels_meta=["Channel 1", "Channel 2"][: shape[0]],
        levels=levels,
        pixelsize=(0.65, 0.65),
        chunks=chunks,
        shards=shards,
        compressors=compressors,
        # NGFF 0.4 maps to zarr format 2, which cannot shard.
        ngff_version=ngff_version,
        overwrite=True,
    )
    container.get_image(path="0").set_array(patch=synthetic(shape))
    return path


def from_spec(root: Path, spec: ImageSpec) -> Path:
    """Create (or reuse) the fixture a named `ImageSpec` describes."""
    from ngio_benchmarks.core.images import zarr_compressors

    compressors = "auto" if spec.compressors == "auto" else zarr_compressors(spec)
    return image_fixture(
        root,
        spec.shape,
        chunks=spec.chunks,
        shards=spec.shards,
        compressors=compressors,
        levels=spec.levels,
        ngff_version=spec.ngff_version,
    )


def segmentation(n_labels: int, size: int = 512) -> np.ndarray:
    """A label image holding `n_labels` square, non-touching labels."""
    seg = np.zeros((size, size), dtype=np.uint16)
    side = int(np.ceil(np.sqrt(n_labels)))
    step = max(size // side, 2)
    label = 0
    for r in range(side):
        for c in range(side):
            label += 1
            if label > n_labels:
                return seg
            seg[r * step : r * step + step - 1, c * step : c * step + step - 1] = label
    return seg


def roi_frame(n: int):
    """A dataframe in the shape a v1 ROI table backend hands to ngio."""
    import pandas as pd

    return pd.DataFrame(
        {
            "FieldIndex": [f"roi_{i}" for i in range(n)],
            "x_micrometer": [float(i) for i in range(n)],
            "y_micrometer": [float(i) for i in range(n)],
            "z_micrometer": [0.0] * n,
            "len_x_micrometer": [10.0] * n,
            "len_y_micrometer": [10.0] * n,
            "len_z_micrometer": [1.0] * n,
        }
    ).set_index("FieldIndex")
