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


class Result(NamedTuple):
    """One measured case."""

    case: Case
    seconds: float
    peak_mb: float
    note: str = ""
    status: str = OK
    extra: dict[str, str] = {}  # noqa: RUF012

    @classmethod
    def blank(cls, case: Case, status: str, note: str = "") -> Result:
        """A case with no number, and the reason why."""
        return cls(case, float("nan"), float("nan"), note, status)


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
) -> tuple[float, float]:
    """Return (median seconds, peak MiB allocated) for `fn`.

    Runs `fn` exactly `executions(repeats, warmup)` times.

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
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)

    tracemalloc.start()
    try:
        peaks = []
        # Never more allocation passes than the caller asked for timed runs.
        # Building a pyramid at `repeats = 1` should cost one pyramid per phase,
        # not one timed and two more nobody mentioned.
        for _ in range(max(1, min(mem_repeats, repeats))):
            tracemalloc.reset_peak()
            fn()
            peaks.append(tracemalloc.get_traced_memory()[1])
    finally:
        tracemalloc.stop()

    return statistics.median(times), max(peaks) / MB
