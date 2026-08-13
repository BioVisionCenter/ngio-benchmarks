"""If not `da.store`, then what? Every way to get a lazy array into a zarr store.

The harness behind `reports/lazy-array-write-alternatives.md`. Its two companion
reports established what is *wrong* with `da.store` -- it issues one
`__setitem__` per dask block, so a block smaller than zarr's write unit forces a
read-modify-write, and concurrent ones lose updates. This one asks the next
question, which is what to do instead, and it exists because the answer differs
per operation shape and nobody should have to guess which.

    uv run python reports/lazy_write_strategies.py --quick
    uv run python reports/lazy_write_strategies.py
    uv run --with tensorstore --with zarrs --with cubed \
           python reports/lazy_write_strategies.py

Three shapes are measured, each standing for one real caller:

  A   materialize   a lazy array -> a store        ome-zarr-converters-tools
  B1  reduce        zarr -> zarr, fan-in over z    fractal-tasks-core projection
  B2  resample      zarr -> zarr, needs a halo     ngio consolidate / zoom

Every strategy is one module-level function taking plain arguments and returning
None -- no timing, no tally, no argparse inside it. That is deliberate and it is
the point: `--emit-snippets` prints those bodies verbatim, and the report's code
examples are that output rather than a retyping of it. An example that has
drifted from the arm that produced the number beside it is worse than no example
at all.

Deliberately not under `ngio_benchmarks/`, for the same reason as
`dask_store_lock_rmw.py`: everything in that package must import inside every
peer library's environment, and this imports ngio, dask and zarr's internals at
once. The instrument the two share lives in `_probe.py`.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from math import prod
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import dask
import dask.array as da
import numpy as np
import zarr
from dask.utils import SerializableLock

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ngio_benchmarks.core.images import BUILTIN
from reports._probe import (
    MB,
    Tally,
    create,
    fmt,
    marked,
    region,
    store_bytes,
    table,
    write_unit,
)

#: The lock ngio installs process-wide (`ngio/common/_locks.py`). Reconstructed
#: rather than imported so an arm labelled "ngio today" keeps meaning that even
#: if ngio drops it -- this file is a record of what was measured.
NGIO_LOCK = SerializableLock()

#: Codec pipeline replacements, as `compare/_pipeline.py` names them.
ZARRS_PIPELINE = "zarrs.ZarrsCodecPipeline"


# --------------------------------------------------------------------------
# Geometry the strategies share
#
# All of it works in absolute target coordinates and converts to patch-relative
# coordinates only at the point of indexing the patch. Getting that backwards is
# the bug this whole file is about, so it is done once, here.
# --------------------------------------------------------------------------


def bounds(index: Sequence[slice], shape: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """`(start, stop)` per axis, with `None`s resolved against `shape`."""
    out = []
    for sl, extent in zip(index, shape, strict=True):
        start = 0 if sl.start is None else sl.start
        stop = extent if sl.stop is None else min(sl.stop, extent)
        out.append((start, stop))
    return tuple(out)


def unit_grid(
    index: Sequence[slice], shape: Sequence[int], unit: Sequence[int]
) -> Iterator[tuple[slice, ...]]:
    """The target's write-unit grid, clipped to `index`, in absolute coordinates.

    Yields one slice tuple per write unit the region touches, each already
    trimmed to the region. Two properties every caller depends on: the pieces
    are **disjoint**, so threads writing different ones cannot race; and there
    is at most **one per unit**, so a region that does not start on a unit
    boundary costs one read-modify-write of the edge units rather than one per
    dask block.
    """
    per_axis = []
    for (start, stop), size in zip(bounds(index, shape), unit, strict=True):
        edges = []
        edge = (start // size) * size
        while edge < stop:
            edges.append((max(edge, start), min(edge + size, stop)))
            edge += size
        per_axis.append(edges)
    for combo in itertools.product(*per_axis):
        yield tuple(slice(lo, hi) for lo, hi in combo)


def local(piece: Sequence[slice], index: Sequence[slice], shape: Sequence[int]):
    """An absolute slice tuple, rebased onto a patch that covers `index`."""
    return tuple(
        slice(sl.start - start, sl.stop - start)
        for sl, (start, _) in zip(piece, bounds(index, shape), strict=True)
    )


def split_aligned(
    index: Sequence[slice], shape: Sequence[int], unit: Sequence[int]
) -> tuple[tuple[slice, ...] | None, list[tuple[slice, ...]]]:
    """Split a region into its unit-aligned interior and the leftover faces.

    The interior is the largest box inside the region whose every face lands on
    a write-unit boundary, so writing it can never read-modify-write anything.
    The faces are what is left, peeled one axis at a time so they come out
    disjoint -- at most two per axis, and each one at most a unit thick.

    Returns `(None, [whole region])` when no aligned interior exists, which is
    the honest answer for a region narrower than one unit rather than a
    degenerate empty box.
    """
    box = bounds(index, shape)
    interior = []
    for (start, stop), size in zip(box, unit, strict=True):
        lo = -(-start // size) * size
        hi = (stop // size) * size
        if lo >= hi:
            return None, [tuple(slice(a, b) for a, b in box)]
        interior.append((lo, hi))

    faces, current = [], list(box)
    for axis, (lo, hi) in enumerate(interior):
        rest_lo, rest_hi = current[axis]
        if rest_lo < lo:
            faces.append((*current[:axis], (rest_lo, lo), *current[axis + 1 :]))
        if hi < rest_hi:
            faces.append((*current[:axis], (hi, rest_hi), *current[axis + 1 :]))
        current[axis] = (lo, hi)

    return (
        tuple(slice(a, b) for a, b in interior),
        [tuple(slice(a, b) for a, b in face) for face in faces],
    )


def threaded(work: Callable[[Any], None], items: Sequence[Any], workers: int) -> None:
    """Run `work` over `items`, in a pool or inline when `workers == 1`.

    Inline at one worker rather than a one-thread pool, so the sequential arm
    measures sequential writing and not pool overhead.
    """
    if workers == 1:
        for item in items:
            work(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, items))


# --------------------------------------------------------------------------
# Shape A -- materialize: a lazy array into a store
#
# Signature for every arm: (patch, array, index, workers) -> None.
# `patch` is a dask array covering exactly `index`.
# --------------------------------------------------------------------------


def a_numpy(patch: da.Array, array: zarr.Array, index, workers: int) -> None:
    """Compute the whole thing, hand zarr one array. The ceiling, and the RAM bill."""
    array[index] = patch.compute()


def a_store_locked(patch: da.Array, array: zarr.Array, index, workers: int) -> None:
    """What ngio does today: one `__setitem__` per block, serialised by a lock."""
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(patch, array, regions=tuple(index), lock=NGIO_LOCK)


def a_store_unlocked(patch: da.Array, array: zarr.Array, index, workers: int) -> None:
    """What dask does untold. The known-bad control, not a candidate."""
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(patch, array, regions=tuple(index), lock=False)


def a_store_unit(patch: da.Array, array: zarr.Array, index, workers: int) -> None:
    """Rechunk to the write unit first, then store lock-free.

    One block per unit means every write covers a whole unit, so zarr skips the
    read entirely and there is nothing for a lock to protect. `rechunk` is a
    graph operation and costs nothing at call time.
    """
    unit = write_unit(array)
    if patch.chunksize != unit:
        patch = patch.rechunk(unit)
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(patch, array, regions=tuple(index), lock=False)


def a_split(patch: da.Array, array: zarr.Array, index, workers: int) -> None:
    """Unit-aligned interior lock-free in parallel; the unaligned faces serially.

    What `a_store_unit` cannot do alone: when the region does not start on a
    unit boundary, no rechunking makes the edge blocks complete. Splitting them
    off leaves an interior that is provably race-free at any worker count, in
    any process, and confines the read-modify-write to faces one unit thick.
    """
    unit = write_unit(array)
    interior, faces = split_aligned(index, array.shape, unit)
    if interior is not None:
        inner = patch[local(interior, index, array.shape)].rechunk(unit)
        with dask.config.set(scheduler="threads", num_workers=workers):
            da.store(inner, array, regions=tuple(interior), lock=False)
    for face in faces:
        array[face] = patch[local(face, index, array.shape)].compute()


def a_eager(patch: da.Array, array: zarr.Array, index, workers: int) -> None:
    """Compute one write unit, write it, move on. Memory bounded at one unit."""
    for piece in unit_grid(index, array.shape, write_unit(array)):
        array[piece] = patch[local(piece, index, array.shape)].compute(
            scheduler="synchronous"
        )


def a_eager_threads(patch: da.Array, array: zarr.Array, index, workers: int) -> None:
    """`a_eager` over a thread pool. Lock-free by construction: the units are disjoint.

    Each unit computes single-threaded (`scheduler="synchronous"`) because the
    parallelism is the outer pool; nesting dask's own pool inside it would
    oversubscribe the machine and measure the contention rather than the design.
    """

    def write_one(piece: tuple[slice, ...]) -> None:
        array[piece] = patch[local(piece, index, array.shape)].compute(
            scheduler="synchronous"
        )

    threaded(write_one, list(unit_grid(index, array.shape, write_unit(array))), workers)


def a_eager_batched(
    patch: da.Array, array: zarr.Array, index, workers: int, batch: int = 8
) -> None:
    """Compute units in batches through one graph, then write them.

    Recovers what per-unit `.compute()` throws away: when one source tile feeds
    several output units -- the converter's normal case -- a shared graph loads
    it once instead of once per unit.
    """
    pieces = list(unit_grid(index, array.shape, write_unit(array)))
    for start in range(0, len(pieces), batch):
        group = pieces[start : start + batch]
        blocks = dask.compute(
            *[patch[local(piece, index, array.shape)] for piece in group],
            scheduler="threads",
            num_workers=workers,
        )
        for piece, block in zip(group, blocks, strict=True):
            array[piece] = block


def a_tensorstore(patch: da.Array, array: zarr.Array, index, workers: int) -> None:
    """Hand the computed array to tensorstore, which does its own chunk-aligned IO."""
    import tensorstore as ts

    handle = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(array.store.root)},
        },
        open=True,
    ).result()
    handle[tuple(index)].write(patch.compute()).result()


def a_cubed(patch: da.Array, array: zarr.Array, index, workers: int) -> None:
    """A bounded-memory, scheduler-free dask replacement, using its own store call."""
    import cubed

    source = cubed.from_array(patch.compute(), chunks=write_unit(array))
    cubed.store([source], [array], regions=[tuple(index)])


A_STRATEGIES: dict[str, dict[str, Any]] = {
    "numpy": {"fn": a_numpy},
    "store/locked": {"fn": a_store_locked},
    "store/unlocked": {"fn": a_store_unlocked, "control": True},
    "store/unit": {"fn": a_store_unit},
    "split": {"fn": a_split},
    "eager": {"fn": a_eager},
    "eager/threads": {"fn": a_eager_threads},
    "eager/batched": {"fn": a_eager_batched},
    "tensorstore": {"fn": a_tensorstore, "needs": "tensorstore", "counts": False},
    "cubed": {"fn": a_cubed, "needs": "cubed"},
}


# --------------------------------------------------------------------------
# Shape B1 -- reduce: zarr -> zarr, fanning in over z
#
# Signature: (source, target, z_axis, workers) -> None.
# --------------------------------------------------------------------------


def b1_numpy(source: zarr.Array, target: zarr.Array, z: int, workers: int) -> None:
    """Read it all, reduce, write once. Unbounded memory; the ceiling."""
    target[...] = np.expand_dims(source[...].max(axis=z), axis=z)


def b1_store_locked(
    source: zarr.Array, target: zarr.Array, z: int, workers: int
) -> None:
    """What fractal-tasks-core does today, via ngio's `set_array`."""
    reduced = da.expand_dims(da.from_zarr(source).max(axis=z), axis=z)
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(reduced, target, lock=NGIO_LOCK)


def b1_store_unit(source: zarr.Array, target: zarr.Array, z: int, workers: int) -> None:
    """The same reduction, rechunked to the target's write unit, lock-free."""
    reduced = da.expand_dims(da.from_zarr(source).max(axis=z), axis=z)
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(reduced.rechunk(write_unit(target)), target, lock=False)


def b1_eager_threads(
    source: zarr.Array, target: zarr.Array, z: int, workers: int
) -> None:
    """Per output unit: read that column of the source, reduce it, write it whole.

    The read is the full z extent for one yx tile, so memory is bounded at
    `workers x (z * unit_yx)` regardless of how deep the stack is.
    """
    full = tuple(slice(None) for _ in target.shape)

    def write_one(piece: tuple[slice, ...]) -> None:
        column = list(piece)
        column[z] = slice(None)
        target[piece] = np.expand_dims(source[tuple(column)].max(axis=z), axis=z)

    threaded(
        write_one, list(unit_grid(full, target.shape, write_unit(target))), workers
    )


def b1_iterators_by_chunks(source, target, z: int, workers: int) -> None:
    """The chunkwise engine ngio already ships, asked to express a z-reduction.

    `by_chunks()` shapes its ROIs from `ref_image.chunks`, so on a z-chunked
    input it yields one ROI per z plane -- while a z-reduction needs the whole z
    extent per output tile. Run rather than argued about: whatever this does,
    it is what the API does today, and the report recommends building on it.
    """
    from ngio import ImageProcessingIterator

    iterator = ImageProcessingIterator(
        input_image=source, output_image=target
    ).by_chunks()
    iterator.map_as_numpy(lambda block: block.max(axis=z, keepdims=True))


def b1_iterators_by_yx(source, target, z: int, workers: int) -> None:
    """The same, with the ROI shaping that ought to fit a fan-in reduce.

    `by_yx()` is the iterator's own name for "one ROI per yx plane, whole
    stack", which is the access pattern `b1_eager_threads` hand-rolls.
    """
    from ngio import ImageProcessingIterator

    iterator = ImageProcessingIterator(input_image=source, output_image=target).by_yx()
    iterator.map_as_numpy(lambda block: block.max(axis=z, keepdims=True))


B1_STRATEGIES: dict[str, dict[str, Any]] = {
    "numpy": {"fn": b1_numpy},
    "store/locked": {"fn": b1_store_locked},
    "store/unit": {"fn": b1_store_unit},
    "eager/threads": {"fn": b1_eager_threads},
    "iterators/by_chunks": {"fn": b1_iterators_by_chunks, "ome_zarr": True},
    "iterators/by_yx": {"fn": b1_iterators_by_yx, "ome_zarr": True},
}


# --------------------------------------------------------------------------
# Shape B2 -- resample: zarr -> zarr, with a halo
#
# Signature: (source, target, workers) -> None.
# --------------------------------------------------------------------------


def b2_numpy(source: zarr.Array, target: zarr.Array, workers: int) -> None:
    """As `ngio.common._pyramid._on_disk_numpy_zoom`: read all, zoom, write once."""
    from ngio.common._zoom import numpy_zoom

    target[...] = numpy_zoom(source[...], target_shape=target.shape, order="linear")


def b2_dask_target_chunks(source: zarr.Array, target: zarr.Array, workers: int) -> None:
    """As `_on_disk_dask_zoom` ships today: rechunk to `target.chunks`, locked.

    The arm that tests whether the pyramid path carries the same bug as
    `set_slice_as_dask`. On a sharded target `target.chunks` is *not* the write
    unit, so this rechunk aligns the blocks to the wrong grid.
    """
    from ngio.common._zoom import dask_zoom

    zoomed = dask_zoom(da.from_zarr(source), target_shape=target.shape, order="linear")
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(zoomed.rechunk(target.chunks), target, lock=NGIO_LOCK)


def b2_dask_target_unit(source: zarr.Array, target: zarr.Array, workers: int) -> None:
    """The same, rechunked to `shards or chunks` instead, and lock-free."""
    from ngio.common._zoom import dask_zoom

    zoomed = dask_zoom(da.from_zarr(source), target_shape=target.shape, order="linear")
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(zoomed.rechunk(write_unit(target)), target, lock=False)


def b2_coarsen_target_chunks(
    source: zarr.Array, target: zarr.Array, workers: int
) -> None:
    """As `_on_disk_coarsen` ships today."""
    factors = {
        axis: max(1, round(s / t))
        for axis, (s, t) in enumerate(zip(source.shape, target.shape, strict=True))
    }
    coarse = da.coarsen(np.mean, da.from_zarr(source), factors, trim_excess=True)
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(coarse.rechunk(target.chunks), target, lock=NGIO_LOCK)


def b2_coarsen_target_unit(
    source: zarr.Array, target: zarr.Array, workers: int
) -> None:
    """`_on_disk_coarsen` rechunked to the write unit, lock-free."""
    factors = {
        axis: max(1, round(s / t))
        for axis, (s, t) in enumerate(zip(source.shape, target.shape, strict=True))
    }
    coarse = da.coarsen(np.mean, da.from_zarr(source), factors, trim_excess=True)
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(coarse.rechunk(write_unit(target)), target, lock=False)


#: Extra source pixels read around each output unit, so the interpolation
#: kernel sees the neighbours a global zoom would have seen. Two is enough for
#: linear; a wider kernel needs a wider halo, which is the parameter this
#: strategy trades correctness for.
HALO = 2


def b2_eager_threads(source: zarr.Array, target: zarr.Array, workers: int) -> None:
    """Per output unit: read the matching source box plus a halo, zoom, write the core.

    Bounded memory, lock-free, no read-modify-write, and correct across
    processes. The cost is that tiled resampling is only *approximately* the
    global result -- see the `wrong` column, which counts pixels that differ
    from a whole-array zoom rather than pretending they do not.
    """
    from ngio.common._zoom import fast_zoom

    scale = tuple(t / s for s, t in zip(source.shape, target.shape, strict=True))
    full = tuple(slice(None) for _ in target.shape)

    def write_one(piece: tuple[slice, ...]) -> None:
        box, offset = [], []
        for sl, factor, extent in zip(piece, scale, source.shape, strict=True):
            lo = max(0, int(sl.start / factor) - HALO)
            hi = min(extent, int(np.ceil(sl.stop / factor)) + HALO)
            box.append(slice(lo, hi))
            offset.append(sl.start - round(lo * factor))
        block = fast_zoom(source[tuple(box)], zoom=scale, order=1, mode="nearest")
        core = tuple(
            slice(off, off + (sl.stop - sl.start))
            for off, sl in zip(offset, piece, strict=True)
        )
        target[piece] = block[core]

    threaded(
        write_one, list(unit_grid(full, target.shape, write_unit(target))), workers
    )


B2_STRATEGIES: dict[str, dict[str, Any]] = {
    "numpy": {"fn": b2_numpy},
    "dask/target-chunks": {"fn": b2_dask_target_chunks},
    "dask/target-unit": {"fn": b2_dask_target_unit},
    "coarsen/target-chunks": {"fn": b2_coarsen_target_chunks, "ref": "coarsen"},
    "coarsen/target-unit": {"fn": b2_coarsen_target_unit, "ref": "coarsen"},
    "eager/threads": {"fn": b2_eager_threads},
}


def b2_reference(source_data: np.ndarray, out_shape, kind: str) -> np.ndarray:
    """What this arm's output *should* be, per the resampler it implements.

    `coarsen` is a block mean and `zoom` is a linear interpolation: they are
    both correct downsamplers and they do not agree pixel for pixel. Scoring
    coarsen against a zoom reference produced a five-figure `wrong` count for an
    arm that is not wrong at all, which is precisely the kind of number a report
    should not print. Each family is checked against its own definition, and the
    two are compared as *strategies* rather than as right and wrong answers.
    """
    from ngio.common._zoom import numpy_zoom

    if kind != "coarsen":
        return numpy_zoom(source_data, target_shape=out_shape, order="linear")
    factors = {
        axis: max(1, round(s / t))
        for axis, (s, t) in enumerate(zip(source_data.shape, out_shape, strict=True))
    }
    coarse = da.coarsen(np.mean, da.from_array(source_data), factors, trim_excess=True)
    return coarse.compute().astype(source_data.dtype)


# --------------------------------------------------------------------------
# Running an arm
# --------------------------------------------------------------------------


def fit_unit(unit: tuple[int, ...], shape, chunks) -> tuple[int, ...]:
    """Trim a shard shape to a derived array, keeping it a multiple of the chunk.

    `min(shape, shard)` -- what ngio's `PyramidLevel` does -- can leave a shard
    that is no longer a whole number of chunks, which zarr rejects. Rounding up
    to the enclosing chunk keeps both invariants.
    """
    return tuple(
        min(u, -(-s // c) * c) for u, s, c in zip(unit, shape, chunks, strict=True)
    )


def summarise(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Median wall, worst-case corruption.

    Median for wall because it is a timing. **Worst** for `wrong` because a
    clean trial of a racing arm proves nothing -- one corrupt trial proves lost
    updates, so the maximum is the honest summary and the median would launder
    it.
    """
    worst = max(trials, key=lambda row: row["wrong"])
    return {
        **worst,
        "wall": statistics.median(row["wall"] for row in trials),
        "cpu_wall": statistics.median(row["cpu_wall"] for row in trials),
        "trials": len(trials),
    }


def measure_arm(
    build: Callable[[Tally], tuple[zarr.Array, Callable[[], None], np.ndarray]],
    *,
    counts_reads: bool = True,
    trials: int = 3,
) -> dict[str, Any]:
    """Time one strategy, count its reads, then check what it actually wrote.

    `build` returns `(target, run, expected)` against a fresh store wired to the
    tally it is handed. The order below is the correctness of the instrument and
    is inherited from `dask_store_lock_rmw.py`: **snapshot before the readback**,
    because verification reads are reads.
    """
    rows = []
    for _ in range(trials):
        tally = Tally()
        target, run, expected = build(tally)
        tally.reset()

        start, start_cpu = time.perf_counter(), time.process_time()
        run()
        wall = time.perf_counter() - start
        cpu = time.process_time() - start_cpu

        counted = tally.snapshot()
        tally.enabled = False

        written = target[...]
        diff = written != expected
        rows.append(
            {
                "wall": wall,
                "cpu_wall": cpu / wall if wall else float("nan"),
                **(
                    counted
                    if counts_reads
                    else {"calls": "n/a", "mib": "n/a", "keys": "n/a"}
                ),
                "wrong": int(diff.sum()),
                "total": int(expected.size),
                "maxdiff": int(
                    np.abs(written.astype("int64") - expected.astype("int64")).max()
                ),
                "bytes": store_bytes(target),
            }
        )
        shutil.rmtree(Path(target.store.root), ignore_errors=True)
    return summarise(rows)


def available(strategy: dict[str, Any]) -> str | None:
    """The reason this strategy cannot run here, or `None` if it can."""
    needs = strategy.get("needs")
    if needs is None:
        return None
    try:
        __import__(needs)
    except ImportError:
        return f"{needs} not installed"
    return None


# --------------------------------------------------------------------------
# The three shapes
# --------------------------------------------------------------------------


def run_shape_a(root: Path, spec, op: str, blocks: str, workers: int, trials: int):
    """Materialize a lazy patch into a store, once per strategy."""
    source = marked(spec)
    index = region(spec, op)
    expected_patch = np.ascontiguousarray(source[index])
    unit = spec.shards or spec.chunks
    block = {"unit": unit, "chunks": spec.chunks, "whole": expected_patch.shape}[blocks]

    rows = []
    for name, strategy in A_STRATEGIES.items():
        skip = available(strategy)
        label = f"{name}"
        if skip:
            rows.append({"arm": label, "note": skip, "wall": None})
            continue

        def build(tally: Tally, strategy=strategy, block=block):
            target = create(root / "a.zarr", spec, tally)
            # A full-array arm must compare against the whole array; a region
            # arm against a store whose untouched remainder is still fill value.
            expected = np.zeros(spec.shape, dtype=spec.dtype)
            expected[index] = expected_patch
            patch = da.from_array(expected_patch, chunks=block)
            return (
                target,
                lambda: strategy["fn"](patch, target, index, workers),
                expected,
            )

        try:
            row = measure_arm(
                build, counts_reads=strategy.get("counts", True), trials=trials
            )
        except Exception as error:  # a failed arm is a result
            rows.append({"arm": label, "note": f"{type(error).__name__}: {error}"})
            continue
        rows.append(
            {
                "arm": label,
                **row,
                "blocks": prod(
                    -(-e // b) for e, b in zip(expected_patch.shape, block, strict=True)
                ),
                "block_mb": prod(block) * expected_patch.dtype.itemsize / MB,
            }
        )
    return rows


def ngio_image(path: Path, spec, shape, chunks, shards):
    """A one-level ngio container, so the iterator arms get the objects they need.

    `ImageProcessingIterator` takes ngio `Image`s, not `zarr.Array`s, so these
    arms cannot share the counting store the rest of the harness writes through
    -- ngio wraps it in its own `NgioStore`. Their read counts are therefore
    reported as `n/a` rather than as a zero that would read as "issued none".
    """
    from ngio import create_empty_ome_zarr

    shutil.rmtree(path, ignore_errors=True)
    container = create_empty_ome_zarr(
        store=path,
        shape=tuple(shape),
        axes_names=list(spec.axes),
        levels=1,
        pixelsize=1.0,
        chunks=tuple(chunks),
        shards=tuple(shards) if shards else None,
        dtype=spec.dtype,
        ngff_version="0.5",
        overwrite=True,
    )
    return container.get_image(path="0")


def run_ngio_arm(
    root: Path,
    spec,
    fn,
    source_data,
    expected,
    out_shape,
    out_chunks,
    out_shards,
    z: int,
    workers: int,
    trials: int,
) -> dict[str, Any]:
    """Time one iterator arm against real ngio containers, and check what it wrote."""
    rows = []
    for _ in range(trials):
        try:
            src = ngio_image(
                root / "b1-ngio-src.zarr", spec, spec.shape, spec.chunks, spec.shards
            )
            src.set_array(patch=source_data)
            dst = ngio_image(
                root / "b1-ngio-dst.zarr", spec, out_shape, out_chunks, out_shards
            )
            start, start_cpu = time.perf_counter(), time.process_time()
            fn(src, dst, z, workers)
            wall = time.perf_counter() - start
            cpu = time.process_time() - start_cpu
        except Exception as error:  # a failed arm is a result
            return {"note": f"{type(error).__name__}: {error}"}
        written = dst.get_as_numpy()
        rows.append(
            {
                "wall": wall,
                "cpu_wall": cpu / wall if wall else float("nan"),
                "calls": "n/a",
                "keys": "n/a",
                "mib": "n/a",
                "wrong": int((written != expected).sum()),
                "total": int(expected.size),
                "maxdiff": int(
                    np.abs(written.astype("int64") - expected.astype("int64")).max()
                ),
                "bytes": "n/a",
            }
        )
    summary = summarise(rows)
    summary.pop("bytes", None)
    return summary


def run_shape_b1(root: Path, spec, workers: int, trials: int):
    """Fan-in reduce over z, once per strategy."""
    source_data = marked(spec)
    z = spec.axes.index("z")
    out_shape = tuple(1 if i == z else e for i, e in enumerate(spec.shape))
    out_chunks = tuple(1 if i == z else c for i, c in enumerate(spec.chunks))
    out_shards = (
        fit_unit(
            tuple(1 if i == z else s for i, s in enumerate(spec.shards)),
            out_shape,
            out_chunks,
        )
        if spec.shards
        else None
    )
    expected = np.expand_dims(source_data.max(axis=z), axis=z)

    src_tally = Tally()
    src = create(root / "b1-src.zarr", spec, src_tally)
    src[...] = source_data
    src_tally.enabled = False

    rows = []
    for name, strategy in B1_STRATEGIES.items():
        skip = available(strategy)
        if skip:
            rows.append({"arm": name, "note": skip})
            continue
        if strategy.get("ome_zarr"):
            rows.append(
                {
                    "arm": name,
                    **run_ngio_arm(
                        root,
                        spec,
                        strategy["fn"],
                        source_data,
                        expected,
                        out_shape,
                        out_chunks,
                        out_shards,
                        z,
                        workers,
                        trials,
                    ),
                }
            )
            continue

        def build(tally: Tally, strategy=strategy):
            target = create(
                root / "b1.zarr",
                spec,
                tally,
                shape=out_shape,
                chunks=out_chunks,
                shards=out_shards,
            )
            return target, lambda: strategy["fn"](src, target, z, workers), expected

        try:
            row = measure_arm(
                build, counts_reads=strategy.get("counts", True), trials=trials
            )
        except Exception as error:  # a failed arm is a result
            rows.append({"arm": name, "note": f"{type(error).__name__}: {error}"})
            continue
        rows.append({"arm": name, **row})
    shutil.rmtree(root / "b1-src.zarr", ignore_errors=True)
    return rows


def run_shape_b2(root: Path, spec, workers: int, trials: int):
    """Half-scale resample in yx -- one pyramid level -- once per strategy."""
    source_data = marked(spec)
    out_shape = spec.level_shapes()[1]
    out_chunks = tuple(min(c, e) for c, e in zip(spec.chunks, out_shape, strict=True))
    out_shards = fit_unit(spec.shards, out_shape, out_chunks) if spec.shards else None
    references = {
        kind: b2_reference(source_data, out_shape, kind) for kind in ("zoom", "coarsen")
    }

    src_tally = Tally()
    src = create(root / "b2-src.zarr", spec, src_tally)
    src[...] = source_data
    src_tally.enabled = False

    rows = []
    for name, strategy in B2_STRATEGIES.items():
        skip = available(strategy)
        if skip:
            rows.append({"arm": name, "note": skip})
            continue

        def build(tally: Tally, strategy=strategy):
            target = create(
                root / "b2.zarr",
                spec,
                tally,
                shape=out_shape,
                chunks=out_chunks,
                shards=out_shards,
            )
            expected = references[strategy.get("ref", "zoom")]
            return target, lambda: strategy["fn"](src, target, workers), expected

        try:
            row = measure_arm(
                build, counts_reads=strategy.get("counts", True), trials=trials
            )
        except Exception as error:  # a failed arm is a result
            rows.append({"arm": name, "note": f"{type(error).__name__}: {error}"})
            continue
        rows.append({"arm": name, **row})
    shutil.rmtree(root / "b2-src.zarr", ignore_errors=True)
    return rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

COLUMNS = [
    ("arm", "arm"),
    ("wall", "wall"),
    ("cpu/wall", "cpu_wall"),
    ("store reads", "calls"),
    ("keys", "keys"),
    ("MiB read", "mib"),
    ("store", "bytes"),
    ("wrong", "wrong"),
    ("max diff", "maxdiff"),
    ("note", "note"),
]

A_COLUMNS = [*COLUMNS[:1], ("blocks", "blocks"), ("block MB", "block_mb"), *COLUMNS[1:]]


def show(rows: list[dict[str, Any]], columns=COLUMNS) -> str:
    """Format and tabulate, leaving skipped arms visible as skipped."""
    prepared = []
    for row in rows:
        shown = dict(row)
        if shown.get("wall") is None:
            shown.pop("wall", None)
        if "block_mb" in shown:
            shown["block_mb"] = f"{shown['block_mb']:.1f}"
        prepared.append(shown)
    return table(fmt(prepared), columns)


def probe_iterator_api(root: Path, spec) -> list[dict[str, Any]]:
    """What ngio's ROI shapers actually produce, asked rather than assumed.

    Three questions the report's recommendation to build on `ngio.iterators`
    turns on, and none of them is a timing:

    * does `by_chunks()` follow the **write unit** or merely `chunks`? On a
      sharded output those differ by the factor this whole report is about;
    * does `by_chunks(overlap_xy=...)` -- the only halo the API offers -- yield
      ROIs that **overlap on write**? A read halo is what a resample needs; a
      write halo is two ROIs racing for one unit;
    * how many ROIs come out, which is how many writes the engine will issue.
    """
    from ngio import ImageProcessingIterator

    image = ngio_image(root / "probe.zarr", spec, spec.shape, spec.chunks, spec.shards)
    unit = spec.shards or spec.chunks
    rows = []
    for label, build in (
        ("by_chunks()", lambda it: it.by_chunks()),
        ("by_chunks(overlap_xy=2)", lambda it: it.by_chunks(overlap_xy=2)),
        ("by_yx()", lambda it: it.by_yx()),
    ):
        try:
            iterator = build(
                ImageProcessingIterator(input_image=image, output_image=image)
            )
            rows.append(
                {
                    "arm": label,
                    "rois": len(iterator.rois),
                    "note": (
                        f"chunks={spec.chunks} unit={unit} "
                        f"pixels overlap={iterator.check_if_regions_overlap()} "
                        f"units overlap={iterator.check_if_chunks_overlap()}"
                    ),
                }
            )
        except Exception as error:  # a failed probe is a result
            rows.append({"arm": label, "note": f"{type(error).__name__}: {error}"})
    shutil.rmtree(root / "probe.zarr", ignore_errors=True)
    return rows


def emit_snippets() -> None:
    """Print every strategy body, so the report's examples are lifted not retyped."""
    for shape, registry in (
        ("A -- materialize", A_STRATEGIES),
        ("B1 -- reduce", B1_STRATEGIES),
        ("B2 -- resample", B2_STRATEGIES),
    ):
        print(f"\n{'=' * 70}\n{shape}\n{'=' * 70}")
        for name, strategy in registry.items():
            print(f"\n--- {name} ---\n")
            print(inspect.getsource(strategy["fn"]))


def main(argv: list[str] | None = None) -> int:
    """Run every shape and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="quarter the z depth")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--pipeline",
        choices=["zarr-python", "zarrs"],
        default="zarr-python",
        help="which codec pipeline zarr-python encodes through",
    )
    parser.add_argument(
        "--emit-snippets",
        action="store_true",
        help="print each strategy body verbatim and exit",
    )
    args = parser.parse_args(argv)

    if args.emit_snippets:
        emit_snippets()
        return 0

    if args.pipeline == "zarrs":
        # Installed process-wide, before anything opens an array: the pipeline
        # is chosen at open time, so toggling it around the timed call would
        # time the toggle. Note that zarrs reads the store from Rust, so the
        # read counters below go blind -- a `0` in that column under this flag
        # means "not seen", and the arms are labelled `n/a` where that is known.
        zarr.config.set({"codec_pipeline.path": ZARRS_PIPELINE})

    def shrink(spec):
        if not args.quick:
            return spec
        z = spec.axes.index("z")
        shape = list(spec.shape)
        shape[z] = max(spec.shards[z] if spec.shards else 1, shape[z] // 4)
        return spec._replace(shape=tuple(shape))

    targets = {
        "unsharded": shrink(BUILTIN["medium"]),
        "sharded": shrink(BUILTIN["sharded"]),
    }
    root = Path(mkdtemp(prefix="lazy-write-"))
    print(
        f"# working in {root}, workers={args.workers}, "
        f"trials={args.trials}, pipeline={args.pipeline}\n"
    )

    try:
        print("## A -- materialize: a lazy array into a store\n")
        for name, spec in targets.items():
            for op, blocks in (
                ("write_full", "chunks"),
                ("write_full", "unit"),
                ("write_full", "whole"),
                ("write_roi_straddling", "chunks"),
            ):
                print(f"\n### {name}, {op}, source blocks = {blocks}\n")
                print(
                    show(
                        run_shape_a(root, spec, op, blocks, args.workers, args.trials),
                        A_COLUMNS,
                    )
                )

        print("\n\n## B1 -- reduce: fan-in over z\n")
        for name, spec in targets.items():
            print(f"\n### {name}\n")
            print(show(run_shape_b1(root, spec, args.workers, args.trials)))

        print("\n\n## B2 -- resample: half-scale in yx\n")
        for name, spec in targets.items():
            print(f"\n### {name}\n")
            print(show(run_shape_b2(root, spec, args.workers, args.trials)))

        print("\n\n## API probe -- what ngio's ROI shapers produce\n")
        for name, spec in targets.items():
            print(f"\n### {name}\n")
            print(
                show(
                    probe_iterator_api(root, spec),
                    [("shaper", "arm"), ("ROIs", "rois"), ("note", "note")],
                )
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
