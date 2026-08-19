"""Mapping over an iterator: does fanning it out actually pay, and at what cost?

ngio's own `tests/performance/` already covers the iterators, but it covers them
as *op counts* -- and it pins the parallel scenario's tally **equal** to the
serial one, because store operations are invariant to concurrency by design.
So nothing there can answer which mapper to reach for or how many workers to
give it. Wall clock and peak memory are the only instruments that can.

The `iterator` axis is a list of arms, and they fall into three groups. Read a
row against its own baseline, not against the block's fastest number.

**The scaling pair**, and it is a pair rather than two results:

`features` reduces read-only -- units are built with no setter, nothing is
written and `finalize` never runs -- so its curve is what the pool is
worth on its own. `segmentation` maps and writes, and every writing map ends in
a rebuild of the *whole* pyramid that no pool touches. Its curve is therefore
the same speedup with a fixed serial tail bolted on, and the gap between the two
is Amdahl measured rather than asserted. The `consolidate` block prices that
tail on its own, so the two compose.

**The reconciliation arms**, which are the same shape of question one level up.
A tiled map is only embarrassingly parallel until a tile needs to know what its
neighbour saw, and ngio now has three answers to that, each with a serial tail
of its own:

    halo        grow the read, crop the write        (`with_halo`)
    stitch      halo, then resolve ids across tiles  (`stitch=StitchConfig()`)
    detection   halo, then suppress duplicates       (`detect` + `NmsConfig`)

`halo` is `segmentation` plus a read margin and a `HaloCroppingSetter`, so
`halo - segmentation` is what context costs before anything is reconciled.
`stitch` is `halo` plus a whole pass over the banked bands after the map, so
`stitch - halo` is the reconciliation itself, isolated. Those two arms *do* share
a per-ROI function, so that subtraction is real.

`detection` is the read-only counterpart -- no write at all, and the tail is NMS
over the boxes the margin duplicated -- but it is not a subtraction against
anything, because a detector returns boxes and no other arm's `func` does. Read
it against its own `workers` column: the fan-out scales and the NMS does not.

**The collect arm**, which prices the join rather than the fan-out. `table` runs
`reduce_to_table`: the same fan-out as `features`, then the default `coalesce`
-- a concat and a `set_index` on the calling thread once the last worker is done.

`table - features` is deliberately *not* the claim here, and the difference is
worth being exact about because the obvious reading is wrong. The two arms cannot
share a per-ROI function: `features` returns two floats, while a row that goes
into a table has to carry the id of the object it measures, which is a different
computation and not a heavier version of the same one. So the two arms' serial
rows are not comparable, and the first run of this block duly had `table` come
back *faster* than `features` at the same `work`.

What `table` is readable against is **itself across the `workers` axis**. Its
fan-out parallelises and its join does not, so a curve that flattens earlier than
`features`' is the join, measured -- which is the same instrument the whole block
uses on the pyramid rebuild, applied one level up.

The remaining two arms swap the *iteration unit* rather than adding a tail:
`processing` writes an image instead of a label (`ImageProcessingIterator`), and
`masked` iterates one ROI per object off a masking table instead of a tile grid
(`MaskedSegmentationIterator`), which is the case where the ROI count is a fact
about the data rather than about the chunking.

Peak memory is the other half of the answer: both parallel mappers materialise
the unit list on the dispatching thread and then hold one patch per worker, so
`peak_mb` is what says whether a worker count fits as well as whether it pays.
That reading holds for `threaded` and not for `process`, whose patches are
allocated in a child `tracemalloc` cannot see -- the same blindness the
comparison suites answer with `NATIVE = True`, which this suite has no
equivalent of, so those rows carry it in their note instead.

What the column actually separates, measured, is the read-only arms from the
writing ones and nothing finer: ~100-120 MB for `features` and `table` against
~40-45 MB everywhere else, halo included. A `reduce` holds every ROI's result
until the mapper returns the list; a `map` hands each patch to a setter and lets
it go. The halo's extra bytes are real but live and die inside one unit, so they
show up in the clock and not here.

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

Those numbers are the `segmentation`/`features` pair at ngio `03f46927`, which
predates the reconciliation arms. Those were swept separately, at `97220377`, and
`experiments/iterators-api.toml` carries the table; three results from it are
worth having here, because they are what the arms were added to find out.

The halo costs what it reads and nothing else: +181 to +337 ms serially at this
geometry, and it reaches 2.06x on four threads where `segmentation` reaches
2.17x. Putting context on the read side really does leave the parallelism alone.

The stitch resolve is a serial tail eight times the size of the map it follows
-- `stitch - halo` is +3.5 s -- and it flattens the pool from 2.06x to 1.11x,
with `cpu / wall` falling from 4.21x to 1.25x alongside it. That is the same
shape as the pyramid rebuild the `segmentation`/`features` pair measures, an
order of magnitude larger.

And NMS dominates `detection` even at 64 boxes a tile: 1.15x on four threads,
with `otsu` and `label` landing within 2% of each other. When the `work` axis
stops mattering, what the row measures is no longer the mapped function.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ngio_benchmarks.core.data import synthetic
from ngio_benchmarks.core.measure import MB, Measured, Skip
from ngio_benchmarks.internal.fixtures import image_fixture

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: Chunked at 512 so the ROIs `by_chunks` produces are large enough that a pool
#: is measuring parallelism rather than its own dispatch overhead. Dropping to
#: 256 quadruples the ROI count and visibly *lowers* the speedup, which is a
#: finding the `z` axis can be swept to reproduce -- not a default to ship.
_CHUNKS = (1, 1, 512, 512)

#: The read margin the three reconciliation arms share, so `halo`, `stitch` and
#: `detection` differ by what they do with the overlap and not by how much of it
#: they read. 32 px in xy is a plausible object radius on a 512 tile; `z = 1` is
#: there because the fixture chunks z at 1 and `StitchPlan` refuses a halo that
#: does not cover *every* tiled axis -- two z-tiles would otherwise land in one
#: stitch grid cell and be indistinguishable. Keep z in here even when raising
#: the xy margin.
_HALO = {"x": 32, "y": 32, "z": 1}

AXES = {
    # Open, and dispatched by name like `algorithms.kernel`: these select a code
    # path rather than bundling kwargs, so there is no structure for a closed
    # axis's labels to stand for. Six further values -- `halo`, `stitch`,
    # `detection`, `table`, `processing`, `masked` -- are implemented and out of
    # the default for the same reason `stub` is out of `work`'s: they answer
    # "what does this feature cost", not "which mapper", and putting them here
    # would quadruple a sweep whose question they do not share. See
    # `experiments/iterators-api.toml`, which is where they are swept.
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
    # `auto` is a valid value and is now the mappers' own default: `MaxWorkers`
    # is `int | "auto" | None`, and a config can spell it because `axes._coerce`
    # falls back to the raw string when `int()` fails. It is left out of the
    # default because what it resolves to -- min(32, cpu_count() + 4) -- is a
    # fact about the machine, and a default nobody can read off the config file
    # is a poor one.
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

#: The columns `ObjectDetectionIterator.detect` requires, in the order the box
#: builders fill them. `x_*`/`y_*` are mandatory and `z_*` is both-or-neither on
#: every tile; the patches here are always (z, y, x), so it is always both.
#: `confidence` is what `NmsConfig` ranks by when it suppresses a duplicate --
#: without it the iterator falls back to box volume, which is the same ordering
#: these builders produce anyway, spelled out so the ranking is visible.
_BOX_COLUMNS = (
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "confidence",
)

#: How many boxes a tile is allowed to report, largest first.
#:
#: This is part of the measurement, not a guard rail, and the first run of the
#: `detection` arm is why. `ndimage.label` on a thresholded 512x512 patch of this
#: fixture finds *thousands* of components, and ngio's NMS is a greedy
#: `for detection in ranked: any(... for other in kept)` in Python -- quadratic,
#: interpreted, and on the calling thread. Uncapped, the arm came back at 7.0 s
#: against 61 ms for `segmentation` at the same geometry: a 115x row that is
#: almost entirely one Python loop, and would have been read as "detection is
#: expensive" when what it says is "this detector is not one".
#:
#: A real detector -- YOLO, a maxima finder -- reports tens of boxes per tile,
#: having already run its own per-tile NMS. 64 is that regime, and it is what
#: makes `detection - features` the cost of the cross-tile pass rather than the
#: cost of an unrealistic input to it.
#:
#: Raising it is how to price ngio's NMS on its own; it is a constant rather than
#: an axis because `work` names a per-patch cost model and this is not one.
_MAX_BOXES = 64

# Every function below is module-level, and that is a requirement rather than a
# style choice: `ProcessMapper` is spawn-based and pickles `func` by reference, so
# the lambda every other block in this suite hands to `Measured` cannot cross the
# boundary. The `_WORK` table holds them for the same reason.
#
# There are four shapes, because the arms hand `func` four different things. The
# map and reduce arms get the unit payload as one argument -- a patch for a
# segmentation, an `(image, label, roi)` triple for a feature extractor. The two
# collect arms are unpacked by ngio before they reach the function, so
# `reduce_to_table` wants `(image, label, roi)` as three arguments and `detect`
# wants `(patch, roi)` as two.
#
# Whatever the shape, what travels *back* must be small: a result crossing out of
# a worker process must not be a patch, or the row measures pickling rather than
# iteration.


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


def _boxes(mask) -> dict[str, list]:
    """Bounding boxes of `mask`'s connected components, in patch coordinates.

    The shared tail of every `detect` function: label, then `find_objects`,
    which is a single C pass over the labelled array and returns one slice
    tuple per object. Patches arrive as (z, y, x) -- `channel_selection` squeezes
    the channel away -- so the z bounds are always available and are always
    reported, which is what `detect` requires of a tile once any tile reports
    them.

    `confidence` is the box volume. It is free, given the slices, and it makes
    NMS prefer the tile that saw more of an object over the tile that caught its
    edge, which is the ranking a haloed detection wants.

    Only the `_MAX_BOXES` largest survive -- see that constant for why the cap is
    part of the measurement rather than a convenience.
    """
    from scipy import ndimage

    labelled, count = ndimage.label(mask)
    boxes: dict[str, list] = {name: [] for name in _BOX_COLUMNS}
    if count == 0:
        return boxes
    found = ndimage.find_objects(labelled)
    volumes = [
        (z.stop - z.start) * (y.stop - y.start) * (x.stop - x.start)
        for z, y, x in found
    ]
    order = sorted(range(len(found)), key=lambda i: -volumes[i])[:_MAX_BOXES]
    for index in sorted(order):
        z, y, x = found[index]
        boxes["x_min"].append(int(x.start))
        boxes["x_max"].append(int(x.stop))
        boxes["y_min"].append(int(y.start))
        boxes["y_max"].append(int(y.stop))
        boxes["z_min"].append(int(z.start))
        boxes["z_max"].append(int(z.stop))
        boxes["confidence"].append(float(volumes[index]))
    return boxes


def _means_by_label(image, label) -> dict[str, list]:
    """Per-object means over the *stored* label: the shared tail of `table`.

    The ids come off the label array rather than from a fresh `ndimage.label`,
    and that is the whole reason this arm can build a table at all: a patch-local
    numbering restarts at 1 in every ROI, so the rows could not be tied to the
    objects they measure. The stored ids are global, which is what
    `reduce_to_table`'s default `coalesce` indexes by.

    An object straddling a tile boundary therefore contributes one row per tile
    it touches, with each row measuring only its own part. That is what a tiled
    feature extraction produces before anything groups it, and pretending
    otherwise here would measure a `groupby` this arm is not about.
    """
    from scipy import ndimage

    ids = np.unique(label)
    ids = ids[ids > 0]
    if ids.size == 0:
        return {"label": [], "mean": []}
    means = ndimage.mean(image, label, index=ids)
    return {"label": [int(i) for i in ids], "mean": [float(m) for m in means]}


def _seg_stub(patch):
    """A write-back that costs nothing, for isolating the machinery."""
    return (patch > 0).astype("uint32")


def _feat_stub(triple):
    """Two reductions, likewise near-free."""
    image, label, _roi = triple
    return float(image.mean()), int(label.max())


def _det_stub(patch, _roi) -> dict[str, list]:
    """One box covering the whole tile: a detector that does no detecting.

    Not an empty result, deliberately. An empty one would make NMS, the
    anchoring and the renumbering free, so the arm would price the iteration and
    call it detection.
    """
    z, y, x = patch.shape[-3:]
    return {
        "x_min": [0],
        "x_max": [int(x)],
        "y_min": [0],
        "y_max": [int(y)],
        "z_min": [0],
        "z_max": [int(z)],
        "confidence": [float(x * y * z)],
    }


def _tab_stub(_image, label, _roi) -> dict[str, list]:
    """One row per tile, off the label's first id -- near-free, and non-empty.

    Non-empty matters here in a way it does not elsewhere: the default
    `coalesce` raises when *every* ROI returns zero rows, so a stub that
    measured nothing would take the arm down rather than flatter it.
    """
    ids = np.unique(label)
    ids = ids[ids > 0]
    first = int(ids[0]) if ids.size else 1
    return {"label": [first], "mean": [0.0]}


def _seg_otsu(patch):
    """Threshold at Otsu's level: a GIL-bound segmentation."""
    return (patch > _otsu_level(patch)).astype("uint32")


def _feat_otsu(triple):
    """The same GIL-bound scan, reduced instead of written."""
    image, _label, _roi = triple
    level = _otsu_level(image)
    return level, float((image > level).mean())


def _det_otsu(patch, _roi) -> dict[str, list]:
    """Boxes around what the GIL-bound threshold found."""
    return _boxes(patch > _otsu_level(patch))


def _tab_otsu(image, label, _roi) -> dict[str, list]:
    """Per-object means, gated on the GIL-bound scan.

    The threshold is what makes this the `otsu` variant rather than a second
    copy of `label`: the histogram pass runs per ROI and holds the GIL, and only
    the objects surviving it are measured.
    """
    level = _otsu_level(image)
    return _means_by_label(image, np.where(image > level, label, 0))


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


def _det_label(patch, _roi) -> dict[str, list]:
    """Boxes around the connected components: the GIL-releasing detector."""
    return _boxes(patch > patch.mean())


def _tab_label(image, label, _roi) -> dict[str, list]:
    """Per-object means straight off the stored label, no gating scan."""
    return _means_by_label(image, label)


class _Work(NamedTuple):
    """The four shapes one `work` value takes, one per arm family.

    A `work` label names a *cost model* -- how much CPU per patch, and whether it
    holds the GIL -- and every arm has to be able to pay it, or the `work` axis
    would silently mean something different in each half of the table.
    """

    #: `patch -> patch`, for the arms that map and write.
    segment: Callable
    #: `(image, label, roi) -> small`, for the read-only reduce.
    feature: Callable
    #: `(patch, roi) -> boxes`, for `ObjectDetectionIterator.detect`.
    detect: Callable
    #: `(image, label, roi) -> columns`, for `reduce_to_table`.
    table: Callable


#: `work` label -> the four functions that spend it.
_WORK = {
    "stub": _Work(_seg_stub, _feat_stub, _det_stub, _tab_stub),
    "otsu": _Work(_seg_otsu, _feat_otsu, _det_otsu, _tab_otsu),
    "label": _Work(_seg_label, _feat_label, _det_label, _tab_label),
}

#: Why each `work` value is in the sweep, carried onto every row it produces. A
#: curve that does not scale is a fact about the function before it is a fact
#: about the mapper, and a reader should not have to know which is which.
_WORK_NOTES = {
    "stub": "func is ~free",
    "otsu": "func holds the GIL",
    "label": "func releases the GIL",
}

#: What each arm does beyond iterating, in the row's own words. `_note` reads
#: this rather than deriving it, because "read-only" and "+ consolidate" are
#: claims about which serial tail the number contains and those do not follow
#: from the arm's name.
_ARM_NOTES = {
    "segmentation": "+ consolidate",
    "features": "read-only",
    "halo": "+ halo crop, + consolidate",
    "stitch": "+ halo crop, + stitch resolve, + consolidate",
    "detection": "read-only, + halo, + nms",
    "table": "read-only, + coalesce",
    "processing": "image out, + consolidate",
    "masked": "masking rois, + consolidate",
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
    the shipped `z` on a tiled arm, but `masked` takes its ROI count from the
    masking table rather than from the chunk grid, so it is one comparison
    against the day either of those changes.

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


def _mask_labels(shape: tuple[int, ...], tile: int) -> np.ndarray:
    """One label per (z-plane, xy tile), which is one *chunk* per masking ROI.

    The alignment is load-bearing rather than tidy. `MaskedSegmentationIterator`
    takes its ROIs from the masking table, so their write footprints are whatever
    the objects happen to be -- and both parallel mappers *raise* on ROIs sharing
    a write unit rather than degrading to serial. Objects laid out on the chunk
    grid are the case where a masked map parallelises at all; a mask whose
    objects straddle chunks would turn every `threaded` row on this arm into a
    stack trace, which is a fact about that mask and not about ngio.

    It also lands the arm on the same ROI count as the tiled arms at the same
    `z`, so `masked` reads against `segmentation` as one changed variable.
    """
    depth, height, width = shape[-3:]
    rows, columns = height // tile, width // tile
    plane = np.repeat(
        np.repeat(
            np.arange(rows * columns, dtype="uint16").reshape(rows, columns),
            tile,
            axis=0,
        ),
        tile,
        axis=1,
    )
    per_plane = rows * columns
    return np.stack([plane + index * per_plane + 1 for index in range(depth)])


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
    # `ProcessMapper`, `map`/`reduce`, `with_halo`, `StitchConfig`,
    # `ObjectDetectionIterator` or `reduce_to_table` exists in ngio 1.0.0, which
    # is what `pyproject.toml`'s `internal` extra resolves to. Inside such an
    # environment this raises and the runner degrades *this block* to one
    # `unavailable` row, leaving the other four to run.
    from ngio.iterators import (
        BasicMapper,
        FeatureExtractorIterator,
        ImageProcessingIterator,
        MaskedSegmentationIterator,
        NmsConfig,
        ObjectDetectionIterator,
        ProcessMapper,
        SegmentationIterator,
        StitchConfig,
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
    if iterator not in _ARM_NOTES:
        raise SystemExit(
            f"iterators has no iterator {iterator!r}; "
            f"choose from: {', '.join(_ARM_NOTES)}"
        )
    funcs = _WORK[work]

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

    # `max_workers=` rather than positional: it is the mappers' only argument,
    # its default is now `"auto"`, and a config may sweep that string straight
    # through `axes._coerce`.
    if mapper == "basic":
        scheduler = BasicMapper()
    elif mapper == "threaded":
        scheduler = ThreadedMapper(max_workers=workers)
    elif mapper == "process":
        scheduler = ProcessMapper(max_workers=workers)
    else:
        raise SystemExit(f"iterators has no mapper {mapper!r}")

    megabytes = math.prod(shape) * 2 / MB

    def measured(built, call) -> Measured:
        """One `Measured`, so every arm notes its ROI count the same way."""
        return Measured(
            call,
            _note(built.rois, megabytes, mapper, workers, work, _ARM_NOTES[iterator]),
        )

    # `channel_selection=0` is not optional on any arm that pairs the image with
    # a label. `derive_label` squeezes the channel axis, so the label is
    # (z, y, x) while an unselected image getter yields (c, z, y, x), and the
    # setter rejects the extra axis outright.
    #
    # `by_write_units()` tiles by the output's write granularity -- the shard
    # shape when the output is sharded, the chunk shape otherwise -- rather than
    # by the input's chunk grid, which is what `by_chunks()` does. On this
    # unsharded fixture the two coincide and it changes nothing; it is here
    # because the parallel mappers *raise* on ROIs sharing a write unit rather
    # than degrading to serial, so the moment a config gives the fixture shards
    # this is the difference between a sweep and a stack trace.
    #
    # The read-only arms keep `by_chunks()`: with no output image there is no
    # write granularity to tile by.
    #
    # `consolidation_mode` is pinned rather than swept: `consolidate` already
    # owns the mode question, and the `None` default changes in 1.2 -- which is
    # why ngio's own scenario pins it too.
    if iterator == "segmentation":
        seg = SegmentationIterator(
            image, label, channel_selection=0, consolidation_mode="dask"
        ).by_write_units()
        return measured(seg, lambda: seg.map(funcs.segment, mapper=scheduler))

    if iterator == "halo":
        # The same arm plus a read margin: the getter reads the grown region and
        # a `HaloCroppingSetter` trims it back before the write, so the ROIs --
        # and therefore the write footprints, and therefore how far this
        # parallelises -- are identical to `segmentation`'s. That is the point of
        # doing context on the read side, and it is what makes the subtraction
        # between the two arms mean only "the halo".
        halo = (
            SegmentationIterator(
                image, label, channel_selection=0, consolidation_mode="dask"
            )
            .with_halo(**_HALO)
            .by_write_units()
        )
        return measured(halo, lambda: halo.map(funcs.segment, mapper=scheduler))

    if iterator == "stitch":
        # `halo` plus the resolve. The map banks each tile's overlap bands as it
        # writes, and `finalize` then walks them, unions the ids that
        # agree across a boundary and compacts the survivors -- all of it serial,
        # all of it after the last worker has finished, and all of it *before*
        # the pyramid rebuild, because every level is derived from level 0.
        #
        # `StitchConfig()` defaults are what a caller gets from `stitch=True`;
        # spelled out because `scratch_store=None` is the one that has to stay
        # None here. An in-memory scratch is often the better choice and cannot
        # cross a process boundary, so pinning it would cost the `process` rows.
        stitch = (
            SegmentationIterator(
                image,
                label,
                channel_selection=0,
                consolidation_mode="dask",
                stitch=StitchConfig(),
            )
            .with_halo(**_HALO)
            .by_write_units()
        )
        return measured(stitch, lambda: stitch.map(funcs.segment, mapper=scheduler))

    if iterator == "detection":
        # Read-only, and the only read-only iterator ngio lets take a halo:
        # there is no write to crop the margin off, so `_allow_readonly_halo`
        # opts it in and NMS is what reconciles the duplicates the margin
        # produces. `by_chunks` rather than `by_write_units` -- with no output
        # image there is no write granularity to tile by.
        det = (
            ObjectDetectionIterator(image, channel_selection=0, nms=NmsConfig())
            .with_halo(**_HALO)
            .by_chunks()
        )
        return measured(det, lambda: det.detect(funcs.detect, mapper=scheduler))

    if iterator == "features":
        feat = FeatureExtractorIterator(image, label, channel_selection=0).by_chunks()
        return measured(feat, lambda: feat.reduce(funcs.feature, mapper=scheduler))

    if iterator == "table":
        # `features` routed through the default `coalesce`: the same fan-out,
        # then one concat and one `set_index` on the calling thread. Nothing is
        # written -- the returned `FeatureTable` is dropped rather than stored,
        # because `add_table` is a different measurement and `roi` already owns
        # the table-write question.
        tab = FeatureExtractorIterator(image, label, channel_selection=0).by_chunks()
        return measured(tab, lambda: tab.reduce_to_table(funcs.table, mapper=scheduler))

    if iterator == "processing":
        # The output is an image rather than a label, which means a second
        # container: `derive_image` writes a new store rather than adding an
        # array to this one. uint32 because the `work` functions all return
        # uint32, and a setter casting on every write would be measuring the
        # cast.
        out = container.derive_image(
            store=root / "processed.zarr", dtype="uint32", overwrite=True
        ).get_image(path="0")
        proc = ImageProcessingIterator(
            image,
            out,
            input_channel_selection=0,
            output_channel_selection=0,
            consolidation_mode="dask",
        ).by_write_units()
        return measured(proc, lambda: proc.map(funcs.segment, mapper=scheduler))

    if iterator == "masked":
        # The one arm whose ROIs are a fact about the data: they come off a
        # masking table, one per object, so the tile grid never enters into it.
        # `get_masked_image` builds the masking table from the label when the
        # container holds none, which is why nothing is added to `tables` here.
        # No `by_chunks` either -- re-tiling these ROIs would put the arm back on
        # the chunk grid and delete the only thing it measures.
        mask = container.derive_label("mask", dtype="uint16", overwrite=True)
        mask.set_array(patch=_mask_labels(mask.shape, _CHUNKS[-1]))
        masked = container.get_masked_image(masking_label_name="mask")
        seg = MaskedSegmentationIterator(
            masked, label, channel_selection=0, consolidation_mode="dask"
        )
        return measured(seg, lambda: seg.map(funcs.segment, mapper=scheduler))

    raise SystemExit(f"iterators has no iterator {iterator!r}")
