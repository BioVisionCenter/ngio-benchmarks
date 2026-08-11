"""Running the internal blocks. Shared by the in-process path and a child.

An environment is either the interpreter you invoked (no `[[environments]]`) or
a `uv` environment holding a pinned ngio. Both end up here, which is what keeps
the two paths from drifting: `[[environments]]` is not a second runner, it is
the same runner re-executed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ngio_benchmarks.core.axes import Case, apply_overrides, cases, count
from ngio_benchmarks.core.config import DEFAULT_REPEATS
from ngio_benchmarks.core.measure import (
    UNAVAILABLE,
    Measured,
    Result,
    Skip,
    executions,
    load_average,
    measure,
    process_peak_mb,
)
from ngio_benchmarks.core.output import (
    MEASUREMENT_FIELDS,
    Schema,
    interpreter,
    log,
    version,
)
from ngio_benchmarks.internal import BLOCKS

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.axes import Override

#: Axis names across every block, as a literal.
#:
#: Deliberately *not* computed by importing the blocks: inside an older
#: environment a block is allowed to fail to import, which would give a child a
#: narrower header than its parent and a merge that `KeyError`s. Six names
#: against four blocks is cheap to maintain, a stale entry costs one empty
#: column, and `--list` makes a mismatch obvious.
AXIS_FIELDS = ("alignment", "kernel", "layout", "mode", "n", "size", "z")

SCHEMA = Schema(
    fields=(
        "env",
        "ngio_version",
        "python",
        "zarr",
        "platform",
        "block",
        "case",
        *AXIS_FIELDS,
        *MEASUREMENT_FIELDS,
        "status",
        "note",
    ),
    axis_fields=AXIS_FIELDS,
    column="env",
)


def environment(label: str) -> dict[str, str]:
    """Describe the interpreter and libraries these numbers came from.

    Recorded on every row, not once per file. Installing an older ngio also
    resolves *its* dependency versions, so a cross-environment difference can
    come from zarr rather than from ngio. These columns are what make a
    surprise attributable instead of a mystery -- check them before concluding
    anything about an ngio change.
    """
    return {
        "env": label,
        "ngio_version": version("ngio"),
        "zarr": version("zarr"),
        # Still per process here, unlike the comparison suites: the internal
        # blocks share fixtures and an interpreter, so this is the high-water
        # mark across all of them. `peak_mb` is the per-case column.
        "proc_peak_mb": f"{process_peak_mb():.1f}",
        "loadavg": f"{load_average():.2f}",
        **interpreter(),
    }


def load(selected: list[str]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Import the selected block modules; collect the ones that would not.

    A block that cannot import must not take discovery down with it -- inside a
    pinned older ngio, something missing is expected -- so this returns the
    failures rather than raising them.
    """
    modules: dict[str, Any] = {}
    unavailable: list[tuple[str, str]] = []
    for name in selected:
        try:
            modules[name] = importlib.import_module(BLOCKS[name])
        except ImportError as error:
            unavailable.append((name, str(error)))
    return modules, unavailable


def resolve(
    modules: dict[str, Any], overrides: tuple[Override, ...]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Apply every override, eagerly, before any measuring starts.

    A bad closed-axis label should not surface after the first block has
    already run -- still less after six environments have been installed.
    """
    for target, _, _ in overrides:
        if target not in modules:
            raise SystemExit(
                f"[axes.{target}] names a block that is not selected; "
                f"selected: {', '.join(modules) or 'none'}"
            )
    return {
        name: apply_overrides(name, getattr(module, "AXES", {}), overrides)
        for name, module in modules.items()
    }


def run_blocks(
    modules: dict[str, Any],
    resolved: dict[str, dict[str, dict[str, Any]]],
    root: Path,
    *,
    repeats: int | None,
    warmup: int,
    quiet: bool,
) -> tuple[list[Result], int]:
    """Measure every case of every module; return the results and skip count."""
    results: list[Result] = []
    skipped = 0
    total = len(modules)
    for index, (name, module) in enumerate(modules.items(), start=1):
        if not quiet:
            log(f"[{index}/{total}] running {name} ...")
        repeat = repeats or getattr(module, "REPEATS", DEFAULT_REPEATS)
        for case in cases(name, resolved[name]):
            try:
                measured: Measured = module.run(root, **case.values)
            except Skip:
                skipped += 1
                continue
            except ImportError as error:
                # Private paths this block needs are gone: degrade the block,
                # not the run, and stop retrying it case by case.
                results.append(Result.blank(case, UNAVAILABLE, f"unavailable: {error}"))
                break
            timing = measure(
                measured.fn, repeats=repeat, warmup=warmup, setup=measured.setup
            )
            results.append(
                Result.timed(case, timing, measured.note, dict(measured.extra))
            )
    return results, skipped


def unavailable_results(unavailable: list[tuple[str, str]]) -> list[Result]:
    """One blank row per block that would not import."""
    return [
        Result.blank(Case(name, {}, {}), UNAVAILABLE, f"unavailable: {error}")
        for name, error in unavailable
    ]


def print_list(
    modules: dict[str, Any],
    resolved: dict[str, dict[str, dict[str, Any]]],
    unavailable: list[tuple[str, str]],
    repeats: int | None,
    warmup: int = 1,
) -> None:
    """Print every selected block and the axes it would actually sweep.

    `runs/case` is the real number of times each case's callable is
    executed -- warmup, then the timed loop, then the allocation passes.
    It is not `repeats`, and a long sweep should not have to discover the
    difference from the clock.

    Overrides are applied first, so this is a dry run of the experiment rather
    than a catalogue of the defaults -- the difference between confirming a
    file does what you meant and being told what the blocks happen to declare.
    """
    for name, module in modules.items():
        declared = getattr(module, "AXES", {})
        axes = resolved[name]
        repeat = repeats or getattr(module, "REPEATS", DEFAULT_REPEATS)
        runs = executions(repeat, warmup)
        print(f"\n{name}  ({count(axes)} cases, {repeat} repeats, {runs} runs/case)")
        summary = (module.__doc__ or "").strip().splitlines()
        if summary:
            print(f"  {summary[0]}")
        width = max((len(a) for a in axes), default=0)
        for axis, values in axes.items():
            kind = "closed" if isinstance(declared[axis], dict) else "open"
            print(f"    {axis:<{width}}  [{kind}]  {', '.join(str(v) for v in values)}")
    for name, error in unavailable:
        print(f"\n{name}\n    unavailable: {error}")
