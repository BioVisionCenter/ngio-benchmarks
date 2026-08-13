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


def audit(path: Path, spec: ImageSpec, op: str) -> dict[str, str]:
    """Read a write back off disk and hash what it actually left there.

    This exists because of a row that looked fine. `dask` writing the `sharded`
    image used to post a good time for output that was **wrong**: with its
    write chunked to `spec.chunks` and stored via `da.store(..., lock=False)`,
    its blocks were single chunks while the target's write unit was a
    256-chunk shard, so every block made zarr read-modify-write the whole
    shard, and the lack of a lock let those races lose each other's updates.
    Nothing in the suite was looking -- the checksum was computed for reads
    only -- so the fast number stood unchallenged. Measured in
    `reports/dask-sharded-write-races.md`. `dask`'s write now goes through
    `da.to_zarr`, which rechunks to the write unit and needs no lock, but the
    audit stays: it is what would have caught this the first time, and a
    write path silently starting to corrupt again is exactly the kind of
    regression a benchmark table would not otherwise notice.

    Filed under the same `checksum` column as a read, deliberately, because it is
    the same claim: *the digest of the pixels this row was responsible for*. That
    makes the whole audit apparatus apply unchanged -- a comparison cell is keyed
    on `(op, image)`, so writers are held against writers of the same operation,
    and a row that disagrees with the rest of its cell gets hatched exactly as a
    divergent read does.

    Read back with zarr-python rather than with whichever library wrote it: the
    auditor must not be one of the contestants, or the one writer it could never
    catch is the one it agrees with. `compare.create._ops.audit` refuses the same
    thing for the same reason.

    That costs nothing today -- zarr resolves into all six io environments, so
    every implementation is audited, `z5py` and `tensorstore` included. The guard
    below stays anyway, because `REQUIRES` is the adapter's to declare and an
    environment without zarr is a thing an adapter is allowed to ask for. If one
    ever does, its cell goes blank, and **a blank cell means "not checked", never
    "passed"**: `report._model._fair` returns `None` for it, which the page draws
    differently from a failure.
    """
    try:
        import zarr
    except ImportError:  # tensorstore, z5py -- no zarr, so no audit
        return {}
    from ngio_benchmarks.compare._run import checksum

    if not path.exists():
        return {}
    # ngio writes an OME container whose pixels are at "0"; the peers write a
    # bare array at the root. Which one is not the thing being audited.
    candidates = (
        lambda: zarr.open_array(store=path, path="0"),
        lambda: zarr.open_array(store=path),
    )
    for opened in candidates:
        try:
            array = opened()
        except Exception:
            # Any failure here means "not that shape" -- try the other. A store
            # neither of them opens leaves the cell blank, which reads as
            # unchecked rather than as passing.
            continue
        return {"checksum": checksum(array[region(spec, op)])}
    return {}


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
