"""Where results go: the CSV files and the ASCII tables.

One schema per suite rather than a union across all three. The suites are
separated precisely so they are not read together, and a union header would be
mostly empty columns in every file. Within a suite the old rules hold: the
environment is repeated on every row so a single file answers cross-environment
questions with no joins, runs append so several accumulate, and a file whose
header does not match raises rather than being appended to.
"""

from __future__ import annotations

import csv
import platform
import sys
from typing import TYPE_CHECKING, NamedTuple

from ngio_benchmarks.core.measure import OK

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from ngio_benchmarks.core.measure import Result


class Schema(NamedTuple):
    """One suite's CSV columns.

    `axis_fields` is a hand-maintained literal, deliberately *not* computed by
    importing the blocks: a block is allowed to fail to import inside an older
    or narrower environment, which would give a child a shorter header than its
    parent and a merge that `KeyError`s. A stale entry costs one empty column,
    and `--list` makes a mismatch obvious.
    """

    fields: tuple[str, ...]
    axis_fields: tuple[str, ...]
    #: Which column names the table columns in the comparison view: the
    #: environment for `internal`, the implementation for `compare`.
    column: str = "env"
    #: Which column splits the output into separate tables. The internal suite
    #: prints one per block; the comparison suites print one per operation,
    #: which is what makes a row a like-for-like comparison across columns.
    group: str = "block"


def interpreter() -> dict[str, str]:
    """Describe the interpreter and platform these numbers came from."""
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system().lower(),
    }


def version(package: str) -> str:
    """The installed version of `package`, or an empty string.

    By distribution metadata rather than `module.__version__`, so a peer that
    does not expose one still gets recorded. Which version produced a number is
    not decoration here: a comparison table is worthless without it.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        return _version(package)
    except PackageNotFoundError:
        return ""


def as_row(result: Result, schema: Schema, env: dict[str, str]) -> dict[str, str]:
    """Flatten one result into a CSV row.

    Axis columns hold *labels*, not values: `layout` is the string `sharded`,
    never a repr of a kwargs dict, and `image` is `medium`. Labels are unique
    per axis and are already what a config file speaks, so a CSV cell and a
    config token are the same token, and
    `df.pivot(index="image", columns="impl", values="seconds")` is a one-liner.
    """
    row = dict.fromkeys(schema.fields, "")
    row.update(env)
    row.update(
        {
            schema.group: result.case.block,
            "case": result.case.label,
            "seconds": ""
            if result.seconds != result.seconds
            else f"{result.seconds:.6f}",
            "peak_mb": ""
            if result.peak_mb != result.peak_mb
            else f"{result.peak_mb:.3f}",
            "status": result.status,
            "note": result.note,
        }
    )
    row.update({k: v for k, v in result.case.labels.items() if k in schema.axis_fields})
    row.update({k: v for k, v in result.extra.items() if k in schema.fields})
    return {field: str(row.get(field, "")) for field in schema.fields}


def check_csv(path: Path, schema: Schema) -> bool:
    """Return whether `path` is an appendable results file; raise if it clashes.

    Called once before any measuring starts, so an unusable `csv` costs nothing
    rather than being discovered after a half-hour run and six installs.
    """
    if not (path.exists() and path.stat().st_size > 0):
        path.parent.mkdir(parents=True, exist_ok=True)
        return False
    with path.open(newline="") as handle:
        existing = next(csv.reader(handle), [])
    if tuple(existing) != schema.fields:
        raise SystemExit(
            f"{path} was written with a different set of columns "
            f"({', '.join(existing)}).\nAppending would misalign it -- "
            "write to a new file instead."
        )
    return True


def write_csv(path: Path, schema: Schema, rows: Iterable[dict[str, str]]) -> None:
    """Append rows to `path`, writing a header only for a new file."""
    appending = check_csv(path, schema)
    with path.open("a", newline="") as handle:
        # A child running an older library may emit a narrower row than this
        # parent knows about; `restval` merges it instead of raising.
        writer = csv.DictWriter(
            handle, fieldnames=schema.fields, restval="", extrasaction="ignore"
        )
        if not appending:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read back a results CSV."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def report(results: list[Result]) -> None:
    """Print one table per block, for a single-environment run."""
    if not results:
        print("\nno results\n")
        return
    width = max(len(r.case.label) for r in results)
    current = None
    for r in results:
        if r.case.block != current:
            current = r.case.block
            print(f"\n{current}")
            print(f"  {'case':<{width}} {'median':>11} {'peak RAM':>13}  note")
            print(f"  {'-' * width} {'-' * 11} {'-' * 13}  ----")
        if r.status != OK:
            print(f"  {r.case.label:<{width}} {r.status:>11} {'':>13}  {r.note}")
            continue
        seconds = f"{r.seconds * 1000:>8.1f} ms" if r.seconds == r.seconds else " " * 11
        peak = f"{r.peak_mb:>10.1f} MB" if r.peak_mb == r.peak_mb else f"{'n/a':>13}"
        print(f"  {r.case.label:<{width}} {seconds} {peak}  {r.note}")
    print()


#: Width of one comparison column, matching the data cell rendered below:
#: 10 chars + " ms" + space + 7 chars + " MB".
_COL = 24


def _short(label: str) -> str:
    """Shorten a label for display; the CSV always keeps the full string."""
    return label if len(label) <= _COL else label[: _COL - 1] + "…"


def report_matrix(rows: list[dict[str, str]], schema: Schema) -> None:
    """Print one table per block, a column per environment or implementation.

    The trailing ratio is against the first column that actually produced data.
    A column that failed to install still has rows recording the failure, so it
    must be excluded from that choice or every ratio would be taken against an
    environment that never ran.
    """
    key, group = schema.column, schema.group
    columns: list[str] = []
    blocks: list[str] = []
    for row in rows:
        if row[key] not in columns:
            columns.append(row[key])
        if row[group] not in blocks and row[group] != "-":
            blocks.append(row[group])
    if not blocks:
        print(f"\nno results: every {key} failed to run\n")
        return

    index = {(r[key], r[group], r["case"]): r for r in rows}
    # A column earns the right to anchor the ratio only by producing a number.
    # An environment that failed to install still has rows recording that, and
    # an implementation may be present with every case `unsupported`; ratios
    # against either would be taken against a column that never ran.
    measured = [
        c
        for c in columns
        if any(
            r[key] == c and r[group] != "-" and r.get("status", OK) == OK for r in rows
        )
    ]
    base = measured[0] if measured else columns[0]
    last = measured[-1] if measured else None

    for block in blocks:
        cases: list[str] = []
        for row in rows:
            if row[group] == block and row["case"] not in cases:
                cases.append(row["case"])
        width = max(len(c) for c in cases)
        head = f"vs {_short(base)}"

        print(f"\n{block}")
        print(
            f"  {'case':<{width}}"
            + "".join(f" {_short(c):>{_COL}}" for c in columns)
            + f" {head:>12}"
        )
        print(
            f"  {'-' * width}"
            + "".join(f" {'-' * _COL}" for _ in columns)
            + f" {'-' * 12}"
        )

        for case in cases:
            line = f"  {case:<{width}}"
            seconds: dict[str, float] = {}
            for column in columns:
                row = index.get((column, block, case))
                if row is None:
                    # Not selected by this config. Distinct from a cell that
                    # ran and could not produce a number -- in a comparison
                    # table an unlabelled blank is a claim.
                    line += f" {'—':>{_COL}}"
                    continue
                if row.get("status", OK) != OK or not row["seconds"]:
                    line += f" {row.get('status', 'failed'):>{_COL}}"
                    continue
                seconds[column] = float(row["seconds"])
                # An empty peak means tracemalloc could not account for this
                # implementation, not that it allocated nothing. `n/a` says so;
                # `0.0 MB` would have been a claim.
                peak = (
                    f"{float(row['peak_mb']):>7.1f} MB" if row["peak_mb"] else "    n/a"
                )
                line += f" {seconds[column] * 1000:>10.1f} ms {peak}"
            if base in seconds and last in seconds and last != base:
                line += f" {seconds[last] / seconds[base]:>11.2f}x"
            print(line)
    _report_process_peaks(rows, key)
    print()


def _report_process_peaks(rows: list[dict[str, str]], key: str) -> None:
    """Print the peak RSS of each child process, once, under the tables.

    Per process rather than per case, because that is what it honestly is: the
    OS high-water mark only rises, so a per-case delta is exact the first time
    and reads zero afterwards. One number per column is coarse, but it is the
    only memory figure that exists at all for an implementation whose buffers
    never pass through the Python allocator -- and for those it is the number
    that answers "will this fit".

    Absolute, not a delta: it includes the interpreter and the library's own
    import footprint, which is part of what running that implementation costs.
    """
    peaks: dict[str, str] = {}
    for row in rows:
        value = row.get("proc_peak_mb")
        if value and row[key] not in peaks:
            peaks[row[key]] = value
    if not peaks:
        return
    print("\npeak RSS per process (includes interpreter and imports)")
    width = max(len(name) for name in peaks)
    for name, value in peaks.items():
        print(f"  {name:<{width}}  {float(value):>8.1f} MB")
