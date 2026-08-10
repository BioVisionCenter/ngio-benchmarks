"""The synthetic pixels every measurement is made on.

Compressibility is the point, not realism of appearance. Uniform data
(`np.ones`) compresses ~2000:1 and pure noise 1:1; either would make a
comparison of chunk shapes, codecs, or writers meaningless without looking
obviously wrong. Real microscopy sits near 1.7x because it is spatially
correlated and quantised to relatively few levels, so this upsamples coarse
noise for structure and then quantises to match.

numpy only, deliberately. `scipy.ndimage.zoom` would be the obvious way to
upsample, and the original suite used it -- but this module is imported inside
every comparison environment, and a tensorstore-only or acquire-zarr-only child
has no scipy and no reason to grow one. The bilinear interpolation below is the
same operation written in ten lines of numpy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

#: Tuned against ngio's bundled sample image, which compresses 1.74x and holds
#: 221 distinct values across the full uint16 range. These give the same regime.
_QUANT_LEVELS = 128
_SMOOTHING = 16


def _bilinear(coarse: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a 2D array to `shape`."""
    height, width = coarse.shape
    rows = np.linspace(0, height - 1, shape[0])
    cols = np.linspace(0, width - 1, shape[1])

    r0 = np.floor(rows).astype(np.intp)
    r1 = np.minimum(r0 + 1, height - 1)
    weight = (rows - r0).astype(np.float32)[:, None]
    stretched = coarse[r0] * (1 - weight) + coarse[r1] * weight

    c0 = np.floor(cols).astype(np.intp)
    c1 = np.minimum(c0 + 1, width - 1)
    weight = (cols - c0).astype(np.float32)[None, :]
    return stretched[:, c0] * (1 - weight) + stretched[:, c1] * weight


def synthetic(shape: tuple[int, ...], dtype: str = "uint16", seed: int = 0):
    """A seeded, spatially correlated volume of `shape`.

    One plane is generated and broadcast across the leading axes. That is not a
    shortcut: it keeps the generator O(y*x) rather than O(volume), which matters
    when the same pixels have to be produced identically inside six separate
    environments, and it leaves the compression ratio independent of depth so a
    `z` sweep varies one thing.
    """
    rng = np.random.default_rng(seed)
    plane_shape = (int(shape[-2]), int(shape[-1]))
    coarse = rng.random(size=tuple(max(s // _SMOOTHING, 2) for s in plane_shape))
    plane = _bilinear(coarse.astype(np.float32), plane_shape)
    plane = np.clip(plane + rng.normal(0, 0.02, plane_shape).astype(np.float32), 0, 1)

    info = np.iinfo(np.dtype(dtype)) if np.dtype(dtype).kind in "iu" else None
    if info is None:
        return np.broadcast_to(plane, shape).astype(dtype)
    levels = min(_QUANT_LEVELS, info.max)
    step = (info.max - max(info.min, 0)) // (levels - 1)
    quantised = (plane * (levels - 1)).round() * step
    return np.broadcast_to(quantised, shape).astype(dtype)


def source_path(root: Path, spec: ImageSpec) -> Path:
    """Where the shared pixel source for `spec` lives."""
    return root / f"source_{spec.name}_{'x'.join(str(s) for s in spec.shape)}.npy"


def write_source(root: Path, spec: ImageSpec):
    """Materialise `spec`'s pixels to a `.npy` beside the fixtures, once.

    Comparison children load this rather than regenerating. Two reasons, and
    the second is the real one: it keeps numpy the only requirement a child has
    beyond its own library, and it makes byte-identity across implementations a
    fact about a file rather than a claim about a shared seed.
    """
    path = source_path(root, spec)
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        # `synthetic` broadcasts, so the array is a view over one plane; `save`
        # materialises it. Written to a temp name and moved, so a run
        # interrupted mid-write cannot leave a truncated file that the next one
        # happily loads.
        #
        # The staging name has to end in `.npy`: `np.save` appends the suffix
        # itself when it is missing, and would otherwise write somewhere this
        # never looks.
        staging = path.with_suffix(".partial.npy")
        np.save(
            staging, np.ascontiguousarray(synthetic(spec.shape, spec.dtype, spec.seed))
        )
        staging.replace(path)
    return path


def load_source(root: Path, spec: ImageSpec):
    """Memory-map the shared pixel source, generating it if absent."""
    return np.load(write_source(root, spec), mmap_mode="r")
