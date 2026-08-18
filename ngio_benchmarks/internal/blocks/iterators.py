"""Mapping over an iterator: does fanning it out actually pay, and at what cost?

ngio's own `tests/performance/` already covers the iterators, but it covers them
as *op counts* -- and it pins the parallel scenario's tally **equal** to the
serial one, because store operations are invariant to concurrency by design.
So nothing there can answer which mapper to reach for or how many workers to
give it. Wall clock and peak memory are the only instruments that can.

Two arms, and the pair is the point:

`features` reduces read-only -- units are built with no setter, nothing is
written and `post_consolidate` never runs -- so its curve is what the pool is
worth on its own. `segmentation` maps and writes, and every writing map ends in
a rebuild of the *whole* pyramid that no pool touches. Its curve is therefore
the same speedup with a fixed serial tail bolted on, and the gap between the two
is Amdahl measured rather than asserted. The `consolidate` block prices that
tail on its own, so the two compose.

Peak memory is the other half of the answer: both parallel mappers materialise
the unit list on the dispatching thread and then hold one patch per worker, so
`peak_mb` is what says whether a worker count fits as well as whether it pays.
That reading holds for `threaded` and not for `process`, whose patches are
allocated in a child `tracemalloc` cannot see -- the same blindness the
comparison suites answer with `NATIVE = True`, which this suite has no
equivalent of, so those rows carry it in their note instead.

One thing measured early and then deliberately not measured again: a resolved
pool of one *is* a `BasicMapper`. Both parallel mappers short-circuit to it, so
`basic`, `threaded` at one worker and `process` at one worker are the same code
path, and the first sweep duly returned three rows agreeing to within noise.
`basic` is the one serial row now, and everything else starts at two.

The `work` axis is what stops the rest of it being a lie. A thread pool can only
overlap work that lets go of the GIL, so a sweep whose `func` costs nothing
measures the machinery's ceiling and nothing a caller would meet. Isolated on the
real per-ROI patch, pure CPU and no IO, the three values are genuinely different
animals:

    stub    (patch > 0).astype        0.04 ms/patch   ~free, overlaps nothing
    otsu    numpy histogram scan      0.80 -> 0.95     holds the GIL, and contends
    label   scipy.ndimage.label       1.30 -> 0.22     releases it, 5.9x at eight

What the block then measured is worth stating carefully, because the obvious
prediction from that table is wrong. A GIL-bound `func` does **not** flatten the
pool: `otsu` still reaches 2.98x on eight threads where `label` reaches 3.46x.
The reason is that neither `func` is the majority of a run at this geometry --
against ~170-230 ms of IO, decode and write, `otsu` adds ~110 ms and `label`
~200 ms, so most of what the pool overlaps is the codec either way. The GIL
dampens the speedup; it does not decide it.

Read the pair as cost against scalability, which are separate properties. On the
read-only arm `label` is the more expensive function serially (390 vs 281 ms) and
is still the more expensive one on eight threads (113 vs 94 ms) -- it does not
overtake. What its higher ratio buys is a larger share of its own extra cost
back: 3.46x against 2.98x. A single mapper column cannot show that, and a single
`work` value would have generalised whichever one it happened to be.

`stub` stays available for isolating the machinery, and is out of the default
because a `func` costing 1.5% of the run flatters every mapper equally.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ngio_benchmarks.core.data import synthetic
from ngio_benchmarks.core.measure import MB, Measured, Skip
from ngio_benchmarks.internal.fixtures import image_fixture

if TYPE_CHECKING:
    from pathlib import Path

#: Chunked at 512 so the ROIs `by_chunks` produces are large enough that a pool
#: is measuring parallelism rather than its own dispatch overhead. Dropping to
#: 256 quadruples the ROI count and visibly *lowers* the speedup, which is a
#: finding the `z` axis can be swept to reproduce -- not a default to ship.
_CHUNKS = (1, 1, 512, 512)

AXES = {
    # Open, and dispatched by name like `algorithms.kernel`: these select a code
    # path rather than bundling kwargs, so there is no structure for a closed
    # axis's labels to stand for.
    "iterator": ["segmentation", "features"],
    # Open. `stub` is the third value and is deliberately not a default -- see
    # the note in the module docstring on why a near-free `func` flatters every
    # mapper equally.
    "work": ["otsu", "label"],
    # Open for the same reason. Constructed and passed as `mapper=` rather than
    # spelled `max_workers=`: the two are mutually exclusive, and `ProcessMapper`
    # is only reachable the long way round anyway.
    "mapper": ["basic", "threaded", "process"],
    # Open, so a bigger machine can sweep past 8. `1` belongs to `basic` alone --
    # see the `Skip`s in `run` -- so a pool starts at 2 here.
    #
    # `auto` is a valid value and is what a bare `ThreadedMapper()` gives:
    # min(32, cpu_count() + 4). It is left out of the default because what it
    # resolves to is a fact about the machine, and a default nobody can read off
    # the config file is a poor one.
    "workers": [1, 2, 4, 8],
    # Open, single-valued by default so it costs no cases until a config names
    # it -- the `c`/`t` pattern from `consolidate`. Sweeping it is how to ask
    # whether the pool pays off better once there is more work per run.
    "z": [16],
}

REPEATS = 3


#: Histogram width for `_otsu_level`. The fixture is uint16, and the cost of the
#: scan is linear in this -- which is part of why `otsu` holds the GIL for as long
#: as it does.
_LEVELS = 1 << 16

# Every function below is module-level, and that is a requirement rather than a
# style choice: `ProcessMapper` is spawn-based and pickles `func` by reference, so
# the lambda every other block in this suite hands to `Measured` cannot cross the
# boundary. The `_WORK` table holds them for the same reason.
#
# The `features` half of each pair takes the `(image, label, roi)` triple a
# `FeatureExtractorIterator` yields, and returns something small: a result
# travelling back out of a worker process must not be a patch, or the row
# measures pickling rather than iteration.


def _otsu_level(plane) -> int:
    """Otsu's threshold, written the way it is usually written: as numpy.

    Every step is vectorised and none of it overlaps, because the work is a
    sequence of passes over a 65k-bin histogram and the GIL is held across all of
    them. That is the point of this function, not a defect in it -- plenty of real
    analysis code looks exactly like this.
    """
    hist = np.bincount(plane.reshape(-1), minlength=_LEVELS).astype("float64")
    total = hist.sum()
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(hist.size)) / total
    between = mu[-1] * omega - mu
    spread = omega * (1.0 - omega)
    # Branchless rather than `errstate` + `nanargmax`: a constant patch makes
    # every `spread` zero, and an all-NaN `nanargmax` raises.
    safe = np.where(spread > 0, spread, 1.0)
    variance = np.where(spread > 0, between**2 / safe, -1.0)
    return int(np.argmax(variance))


def _seg_stub(patch):
    """A write-back that costs nothing, for isolating the machinery."""
    return (patch > 0).astype("uint32")


def _feat_stub(triple):
    """Two reductions, likewise near-free."""
    image, label, _roi = triple
    return float(image.mean()), int(label.max())


def _seg_otsu(patch):
    """Threshold at Otsu's level: a GIL-bound segmentation."""
    return (patch > _otsu_level(patch)).astype("uint32")


def _feat_otsu(triple):
    """The same GIL-bound scan, reduced instead of written."""
    image, _label, _roi = triple
    level = _otsu_level(image)
    return level, float((image > level).mean())


def _seg_label(patch):
    """Connected components: what a segmentation actually ends with.

    `scipy.ndimage` drops the GIL in its C kernel, so this is the arm where a
    thread pool has something to overlap. Imported inside the function, like
    every other third-party import in this block.
    """
    from scipy import ndimage

    labelled, _count = ndimage.label(patch > patch.mean())
    return labelled.astype("uint32")


def _feat_label(triple):
    """Per-object means -- real feature extraction, and it releases the GIL too.

    ~860 objects per ROI on the shipped fixture, so the per-object pass is a real
    share of the cost rather than a rounding error on the labelling.
    """
    from scipy import ndimage

    image, _label, _roi = triple
    labelled, count = ndimage.label(image > image.mean())
    if count == 0:
        return 0, 0.0
    means = ndimage.mean(image, labelled, index=np.arange(1, count + 1))
    return count, float(np.nanmean(means))


#: `work` label -> (what the segmentation arm writes, what the feature arm collects).
_WORK = {
    "stub": (_seg_stub, _feat_stub),
    "otsu": (_seg_otsu, _feat_otsu),
    "label": (_seg_label, _feat_label),
}

#: Why each `work` value is in the sweep, carried onto every row it produces. A
#: curve that does not scale is a fact about the function before it is a fact
#: about the mapper, and a reader should not have to know which is which.
_WORK_NOTES = {
    "stub": "func is ~free",
    "otsu": "func holds the GIL",
    "label": "func releases the GIL",
}


def _note(
    rois: list,
    megabytes: float,
    mapper: str,
    workers: int | str,
    work: str,
    kind: str,
) -> str:
    """The row's note, carrying the two things its numbers cannot say themselves.

    A pool never exceeds the unit count, so a `workers` wider than the ROIs
    silently resolves down -- and a flat tail read without that looks like "more
    workers stop helping" when the pool simply stopped growing. It cannot bind at
    the shipped `z`, and it is one comparison against the day someone lowers it.

    `peak_mb` is `tracemalloc`, which sees nothing a worker *process* allocates.
    The suite has no `NATIVE = True` to suppress the column with, so the `process`
    rows say it in words rather than letting a 0.4 MB cell read as thrift.
    """
    parts = [f"{len(rois)} rois, {megabytes:.0f} MB", kind, _WORK_NOTES[work]]
    if isinstance(workers, int) and workers > len(rois):
        parts.append(f"pool resolves to {len(rois)}")
    if mapper == "process":
        parts.append("peak RAM is parent only")
    return ", ".join(parts)


def run(
    root: Path,
    *,
    iterator: str,
    work: str,
    mapper: str,
    workers: int | str,
    z: int,
) -> Measured:
    """Build the image, the output label and the iterator; measure only the map."""
    from ngio import open_ome_zarr_container

    # Kept out of module scope like every other block's ngio import, and here it
    # is load-bearing rather than ceremonial: none of `ThreadedMapper`,
    # `ProcessMapper` or `reduce_as_*` exists in ngio 1.0.0, which is what
    # `pyproject.toml`'s `internal` extra resolves to. Inside such an
    # environment this raises and the runner degrades *this block* to one
    # `unavailable` row, leaving the other four to run.
    from ngio.iterators import (
        BasicMapper,
        FeatureExtractorIterator,
        ProcessMapper,
        SegmentationIterator,
        ThreadedMapper,
    )

    # Two halves of one fact: serial is measured once, under `basic`.
    #
    # `BasicMapper` takes no worker count, and a parallel mapper whose pool
    # resolves to one short-circuits *to* `BasicMapper` -- ngio's own
    # `ThreadedMapper` docstring says so. Measuring either would put the same
    # code path in three rows under three labels that imply otherwise, which the
    # first sweep of this block confirmed to within noise before these landed.
    if mapper == "basic" and workers != 1:
        raise Skip("basic is serial; a worker count does not reach it")
    if mapper != "basic" and workers == 1:
        raise Skip(f"{mapper} at one worker is BasicMapper; measured as basic")

    if work not in _WORK:
        raise SystemExit(
            f"iterators has no work {work!r}; choose from: {', '.join(_WORK)}"
        )
    segment, extract = _WORK[work]

    shape = (1, z, 1024, 1024)
    path = image_fixture(root, shape, chunks=_CHUNKS)
    container = open_ome_zarr_container(path, mode="r+")
    image = container.get_image(path="0")

    # Derived per case, and filled, for two separate reasons. Filled, because an
    # unwritten zarr array has no chunks on disk and reads back as `fill_value`
    # without touching the store -- which would make the feature arm's label
    # read free and the whole reduce a measurement of the image read alone. And
    # derived afresh, because that is what puts every case, and every repeat
    # within a case, at the same starting state: the first run would otherwise
    # be creating chunks where later ones overwrite them.
    #
    # Generated as uint16 and cast up, rather than asked for as uint32 directly:
    # `synthetic` computes its quantisation in float32, and for uint32 the top
    # level lands at 2**32 once the plane is large enough to reach 1.0 -- an
    # undefined cast that fills the label with garbage and warns once per run.
    # uint16 is the regime that generator is tuned for, and a label holding few
    # distinct values is what a real one holds anyway.
    label = container.derive_label("mapped", dtype="uint32", overwrite=True)
    label.set_array(patch=synthetic(label.shape, "uint16", seed=1).astype("uint32"))

    if mapper == "basic":
        scheduler = BasicMapper()
    elif mapper == "threaded":
        scheduler = ThreadedMapper(workers)
    elif mapper == "process":
        scheduler = ProcessMapper(workers)
    else:
        raise SystemExit(f"iterators has no mapper {mapper!r}")

    megabytes = math.prod(shape) * 2 / MB

    if iterator == "segmentation":
        # `channel_selection=0` is not optional. `derive_label` squeezes the
        # channel axis, so the label is (z, y, x) while the image getter yields
        # (c, z, y, x), and the setter rejects the extra axis outright.
        #
        # `grid="write"` sizes the tiles by the output's write granularity
        # rather than the input's chunk grid. On this unsharded fixture the two
        # coincide and it changes nothing -- it is here because the parallel
        # mappers *raise* on ROIs sharing a write unit rather than degrading to
        # serial, so the moment a config gives the fixture shards this is the
        # difference between a sweep and a stack trace.
        seg = SegmentationIterator(
            image, label, channel_selection=0, consolidation_mode="dask"
        ).by_chunks(grid="write")
        # Pinned rather than swept: `consolidate` already owns the mode
        # question, and the `None` default changes in 1.2 -- which is why ngio's
        # own scenario pins it too.
        note = _note(seg.rois, megabytes, mapper, workers, work, "+ consolidate")
        return Measured(lambda: seg.map_as_numpy(segment, mapper=scheduler), note)

    if iterator == "features":
        feat = FeatureExtractorIterator(image, label, channel_selection=0).by_chunks()
        note = _note(feat.rois, megabytes, mapper, workers, work, "read-only")
        return Measured(lambda: feat.reduce_as_numpy(extract, mapper=scheduler), note)

    raise SystemExit(f"iterators has no iterator {iterator!r}")
