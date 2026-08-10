"""What each of the six operations means, in one place.

Every adapter translates these into its own library's vocabulary, so the
definitions have to live somewhere neutral -- otherwise "read an ROI" quietly
becomes six slightly different reads and the table compares nothing.

The region is taken whole on the leading axes and sliced only on the trailing
two. That matches how ROIs are actually used, and it keeps the aligned and
straddling cases differing in exactly one thing: the offset.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from ngio_benchmarks.core.data import load_source

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

READS = ("read_full", "read_roi_aligned", "read_roi_straddling")
WRITES = ("write_full", "write_roi_aligned", "write_roi_straddling")


def is_read(op: str) -> bool:
    """Whether `op` reads."""
    return op.startswith("read")


def is_full(op: str) -> bool:
    """Whether `op` covers the whole array rather than a region."""
    return op.endswith("_full")


def region(spec: ImageSpec, op: str) -> tuple[slice, ...]:
    """The index for `op`: full axes, then the two sliced trailing ones.

    A tuple of explicit slices rather than `Ellipsis` so that every library
    receives the same shaped selection -- some of them treat `...` differently
    from a full slice, and that difference is not what is being measured.
    """
    if is_full(op):
        return tuple(slice(None) for _ in spec.shape)
    start = spec.roi_offset(aligned=op.endswith("_aligned"))
    stop = start + spec.roi_size
    leading = tuple(slice(None) for _ in spec.shape[:-2])
    return (*leading, slice(start, stop), slice(start, stop))


def patch(spec: ImageSpec, op: str, root: Path):
    """The array a write operation writes.

    Materialised out of the memory-mapped source, because leaving it a memmap
    would fold the read of the source into the timing of the write.
    """
    import numpy as np

    source = load_source(root, spec)
    return np.ascontiguousarray(source[region(spec, op)])


def target(root: Path, impl: str, op: str, spec: ImageSpec) -> Path:
    """A fresh store for a write operation, cleared if it already exists.

    Per implementation and per operation: two writers sharing a path would have
    the second measure an overwrite of the first's chunks, which compresses and
    allocates differently from a write into empty space.
    """
    path = root / f"write_{impl}_{op}_{spec.name}.zarr"
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
