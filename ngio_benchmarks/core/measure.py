"""Measurement itself: what a block hands back, and how it is timed.

A measured module -- a block in `internal`, an adapter in `compare` -- declares
its axes as data and exposes one function that does its own setup and returns
the callable to measure:

    def run(root: Path, **values) -> Measured

Everything outside the returned callable is excluded from the measurement. It
is one function rather than a `(setup, run)` pair because setup and the
measured call always share state.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time
import tracemalloc
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from ngio_benchmarks.core.axes import Case

MB = 1024 * 1024

#: Why a case has no number. In a comparison table a blank cell is a claim, so
#: the three reasons a cell can be empty must stay distinguishable:
#:
#:   ok           measured
#:   unsupported  the implementation declares it cannot do this operation
#:   unavailable  the environment failed to install, or the import failed
#:   failed       it ran and raised
OK = "ok"
UNSUPPORTED = "unsupported"
UNAVAILABLE = "unavailable"
FAILED = "failed"


class Skip(Exception):
    """Raised by a `run` when a case in the product does not apply.

    The cartesian product is rectangular; some blocks are not. `algorithms`
    sweeps three kernels whose useful `n` ranges do not overlap, and an O(n^2)
    kernel at n=50_000 would not return. Raising from inside `run` keeps the
    constraint next to the comment that justifies it, which a product-filter
    hook on the runner would not.
    """


class Unsupported(Exception):
    """Raised by a comparison adapter for an operation it cannot express.

    Distinct from `Skip`: a skipped case was never interesting, an unsupported
    one is a real gap in the matrix and must be printed as such. `acquire-zarr`
    has no read API at all, and `z5py` cannot open zarr v3.
    """


class Measured(NamedTuple):
    """What a block hands back: the callable to time, and an optional note."""

    fn: Callable[[], object]
    note: str = ""
    #: Extra CSV columns this case wants to record -- output store size, the
    #: downsampling filter a writer chose. Values must already be strings.
    extra: dict[str, str] = {}  # noqa: RUF012
    #: Run before every execution of `fn` and excluded from the timing.
    #:
    #: This exists because a writer measured repeatedly does not repeat the same
    #: work: the first run writes into an empty directory and every later one
    #: first has to get rid of the store the previous run left. Each library
    #: pays that differently -- `zarr.open_group(mode="w")`, `overwrite=True`,
    #: an `rmtree` -- so leaving it inside the timed region measures deletion
    #: strategy as much as write speed, and only from the second repeat on.
    #: Clearing the target here means every timed run starts from the same
    #: state, which is the state the first one started from.
    setup: Callable[[], None] | None = None


class Timing(NamedTuple):
    """What one case's timed and allocation phases produced.

    Both a centre and a width, because a single number is not a measurement.
    The centre is the median rather than the mean: with `repeats` in the low
    single digits one interrupted run moves a mean by more than the effect most
    of these comparisons are looking for.
    """

    #: Median of the timed runs, in seconds. The headline number.
    seconds: float
    #: Median absolute deviation of the timed runs. Robust at n=3, where a
    #: sample standard deviation is dominated by whichever run was unluckiest.
    #: Zero at n=1, which the report renders as `n=1` rather than as `+/- 0.0`.
    mad: float
    low: float
    high: float
    #: How many timed runs went into the above -- reported, so a reader can see
    #: when a spread is not estimable rather than inferring it from its width.
    repeats: int
    #: Median CPU time over the same runs. Read against `seconds`: a case whose
    #: wall-clock is well under its CPU time used more than one core, which is
    #: a real difference between these writers and is otherwise invisible.
    cpu_seconds: float
    peak_mb: float


class Result(NamedTuple):
    """One measured case."""

    case: Case
    seconds: float
    peak_mb: float
    note: str = ""
    status: str = OK
    extra: dict[str, str] = {}  # noqa: RUF012
    #: The rest of the timing distribution. Defaulted and placed last so that
    #: `Result(case, seconds, peak, note)` keeps working for the paths that do
    #: not have one.
    mad: float = float("nan")
    low: float = float("nan")
    high: float = float("nan")
    repeats: int = 0
    cpu_seconds: float = float("nan")

    @classmethod
    def blank(
        cls,
        case: Case,
        status: str,
        note: str = "",
        extra: dict[str, str] | None = None,
    ) -> Result:
        """A case with no number, and the reason why.

        `extra` still travels. A row that could not be measured is still a row
        about a particular variant of a particular image, and dropping its
        labels would make `unsupported` read as a fact about the whole
        implementation rather than about the one thing it was asked for.
        """
        return cls(case, float("nan"), float("nan"), note, status, extra or {})

    @classmethod
    def timed(
        cls,
        case: Case,
        timing: Timing,
        note: str = "",
        extra: dict[str, str] | None = None,
    ) -> Result:
        """A case that ran, from what `measure` returned."""
        return cls(
            case,
            timing.seconds,
            timing.peak_mb,
            note,
            OK,
            extra or {},
            timing.mad,
            timing.low,
            timing.high,
            timing.repeats,
            timing.cpu_seconds,
        )


#: How many allocation-tracking passes `measure` makes, at most. Peak is the max
#: over them because the question peak answers is "will this fit", and that is a
#: worst case -- but it is clamped to `repeats`, so a config asking for one run
#: gets one of each rather than three runs of something it did not ask for.
MEM_REPEATS = 2


def process_peak_mb() -> float:
    """Peak resident set size of this process so far, in MiB.

    The OS's own high-water mark, so unlike `tracemalloc` it counts every byte
    -- numpy, but also the buffers a C++ or Rust extension allocates without
    ever telling Python. That is the gap it exists to close: measured on a
    128 MiB read, `tracemalloc` reports 132.5 MB for zarr-python and **0.0 MB**
    for tensorstore, which does the same work in C++.

    It is a *process* number, not a per-case one, and it only ever rises. A
    peak-RSS delta around a single call is exact the first time and reads ~0
    afterwards, because the allocator does not return the pages: measured
    repeatedly on the same 128 MiB read, the first call shows 133.5 MB and
    every later one shows 0.1 MB. So this is recorded once per child process,
    where each child holds one implementation, rather than pretended to be
    per-case.

    Returns 0.0 where `resource` is unavailable (Windows).
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS and the BSDs report bytes.
    return peak / MB if sys.platform == "darwin" else peak * 1024 / MB


def executions(repeats: int, warmup: int) -> int:
    """How many times a case's callable actually runs, given its settings.

    Exported so `--list` can print it. The count is not `warmup + repeats`, and
    a config that asks for one repeat of a job that takes a minute deserves to
    be told that before it starts rather than to work it out from the clock.
    """
    return warmup + repeats + min(MEM_REPEATS, repeats)


def measure(
    fn: Callable[[], Any],
    *,
    repeats: int = 5,
    warmup: int = 1,
    mem_repeats: int = MEM_REPEATS,
    setup: Callable[[], None] | None = None,
) -> Timing:
    """Return the timing distribution and peak allocation for `fn`.

    Runs `fn` exactly `executions(repeats, warmup)` times. `setup`, if given,
    runs before each of those and is excluded from every measurement -- see
    `Measured.setup` for why that is not an optional nicety.

    A `gc.collect()` joins `setup` outside the timed region. Collection is left
    *enabled* during the run: disabling it would flatter exactly the
    implementations that allocate most, which is the comparison several of
    these blocks exist to make.

    Timing and allocation are measured in two separate phases on purpose.
    `tracemalloc` hooks every allocation and inflates allocation-heavy code
    several-fold, and it does not inflate every case equally -- the mode that
    allocates most is penalised most, which is exactly the comparison the
    `consolidate` block exists to make. A single fused loop reports a time that
    is not the time.

    Peak is the max over its runs, not the last one: the question peak answers
    is "will this fit", and that is a worst case.

    What `tracemalloc` does and does not see is worth being exact about, since
    the obvious summary of it is wrong in both directions.

    It **does** see numpy: numpy registers a `tracemalloc` domain and reports
    its data buffers through it. On a 128 MiB read, `tracemalloc` says 132.5 MB
    and the OS says 133.3 MB. For anything whose allocations go through Python
    or numpy -- which is all of the `internal` suite -- this is an absolute
    figure, not merely a relative signal.

    It **does not** see allocations a C++ or Rust extension makes on its own.
    The same 128 MiB read through tensorstore reports 0.0 MB, because the
    buffer never passes through the Python allocator. An implementation that
    declares `NATIVE = True` therefore has this column suppressed rather than
    printed as a zero that reads like "uses no memory"; `proc_peak_mb` is the
    column to read for those.
    """

    def prepare() -> None:
        """Put the world back where the first run found it."""
        if setup is not None:
            setup()
        gc.collect()

    for _ in range(warmup):
        prepare()
        fn()

    times, cpu = [], []
    for _ in range(repeats):
        prepare()
        start, start_cpu = time.perf_counter(), time.process_time()
        fn()
        times.append(time.perf_counter() - start)
        cpu.append(time.process_time() - start_cpu)

    tracemalloc.start()
    try:
        peaks = []
        # Never more allocation passes than the caller asked for timed runs.
        # Building a pyramid at `repeats = 1` should cost one pyramid per phase,
        # not one timed and two more nobody mentioned.
        for _ in range(max(1, min(mem_repeats, repeats))):
            # Before `reset_peak`, so whatever clearing the target allocates is
            # not charged to the case.
            prepare()
            tracemalloc.reset_peak()
            fn()
            peaks.append(tracemalloc.get_traced_memory()[1])
    finally:
        tracemalloc.stop()

    median = statistics.median(times)
    return Timing(
        seconds=median,
        mad=statistics.median([abs(t - median) for t in times]),
        low=min(times),
        high=max(times),
        repeats=len(times),
        cpu_seconds=statistics.median(cpu),
        peak_mb=max(peaks) / MB,
    )


def flush_seconds() -> float:
    """Seconds spent flushing the OS write-back cache, or NaN where it cannot be.

    A timed write ends when the library returns, which on any modern filesystem
    is before the bytes are durable. That is the honest thing to measure -- it
    is what the caller waits for -- but it is not write throughput, and how much
    work is left outstanding differs per writer because their stores differ in
    size. So the flush is timed once, after the timed runs, and reported in its
    own column rather than folded into `seconds`, where it would distort the
    headline while answering a question nobody asked of it.

    System-wide, so on a busy machine it also flushes writes this process never
    made. Read it as an upper bound.
    """
    if not hasattr(os, "sync"):  # pragma: no cover - Windows
        return float("nan")
    start = time.perf_counter()
    os.sync()
    return time.perf_counter() - start


def load_average() -> float:
    """One-minute load average, or NaN where the platform has none.

    Recorded per case so that a row which disagrees with its neighbours can be
    attributed to the machine rather than to the library. Cheap enough that not
    having it is the more expensive option.
    """
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):  # pragma: no cover - Windows
        return float("nan")
