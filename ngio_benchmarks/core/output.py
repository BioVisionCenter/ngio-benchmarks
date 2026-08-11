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
import time
from typing import TYPE_CHECKING, Any, NamedTuple

from ngio_benchmarks.core.measure import OK

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from ngio_benchmarks.core.measure import Result


#: The measurement columns, identical in every suite and spliced into each
#: schema rather than retyped, so the three files stay comparable and a column
#: added here reaches all of them.
#:
#: `seconds` is the median and the rest is what it is a median of. The two
#: memory columns are not interchangeable: `peak_mb` is what `tracemalloc`
#: accounted for inside the case, `proc_peak_mb` is the OS high-water mark for
#: the whole process, and `rss_base_mb` is that mark taken *before* the case ran
#: -- so `proc_peak_mb - rss_base_mb` is the case's cost and `rss_base_mb`
#: alone is what merely loading that library costs.
MEASUREMENT_FIELDS = (
    "seconds",
    "seconds_mad",
    "seconds_min",
    "seconds_max",
    "repeats",
    "cpu_seconds",
    "peak_mb",
    "proc_peak_mb",
    "rss_base_mb",
    "loadavg",
)


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


def log(message: str) -> None:
    """Print a progress line stamped with the wall-clock time, flushed immediately.

    The one place the timestamp format lives -- every progress call site
    routes through this instead of formatting its own prefix.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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

    Total, including on the empty name: a caller asking after something that has
    no package at all -- zarr-python's own codec pipeline, which is not a
    separate install -- wants the same blank cell as a package that is absent,
    and `importlib.metadata` raises a `ValueError` for that one.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    if not package:
        return ""
    try:
        return _version(package)
    except PackageNotFoundError:
        return ""


def _number(value: float, places: int) -> str:
    """Render a float, or the empty string for the NaN that means "no number"."""
    return "" if value != value else f"{value:.{places}f}"


def as_row(result: Result, schema: Schema, env: dict[str, str]) -> dict[str, str]:
    """Flatten one result into a CSV row.

    Axis columns hold *labels*, not values: `layout` is the string `sharded`,
    never a repr of a kwargs dict, and `image` is `medium`. Labels are unique
    per axis and are already what a config file speaks, so a CSV cell and a
    config token are the same token, and
    `df.pivot(index="image", columns="impl", values="seconds")` is a one-liner.

    `seconds` is the median, and the columns beside it are the rest of the
    distribution rather than decoration: `seconds_min`/`seconds_max` bound it,
    `seconds_mad` widths it, and `repeats` says how many runs any of that came
    from. A reader who wants their own estimator has the numbers to build one.
    """
    row = dict.fromkeys(schema.fields, "")
    row.update(env)
    row.update(
        {
            schema.group: result.case.block,
            "case": result.case.label,
            "seconds": _number(result.seconds, 6),
            "seconds_mad": _number(result.mad, 6),
            "seconds_min": _number(result.low, 6),
            "seconds_max": _number(result.high, 6),
            "cpu_seconds": _number(result.cpu_seconds, 6),
            "repeats": str(result.repeats) if result.repeats else "",
            "peak_mb": _number(result.peak_mb, 3),
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


def duration(seconds: float) -> str:
    """A time, in whichever unit keeps it readable at a glance.

    Milliseconds up to a minute and seconds beyond it, rather than one unit
    everywhere: these suites span a chunk read and a pyramid build, and neither
    `0.3 ms` as `0.000` seconds nor a 90-second build as `90000.0 ms` is a
    number anyone can compare by eye.
    """
    if seconds != seconds:
        return ""
    if seconds >= 60:
        return f"{seconds:.1f} s"
    return f"{seconds * 1000:.1f} ms"


def spread(mad: float, low: float, high: float, repeats: int) -> str:
    """The run-to-run width beside a median, or why there is not one.

    Three cases, and they are different claims:

    * `n=1` -- one timed run. There is no spread, and printing `+/- 0.0` would
      assert a precision that was never measured. This is what the shipped
      reference config used to produce for every row.
    * `+/- <half-range>` at n=2, because a median absolute deviation of two
      points is half their gap anyway and the range is the honest name for it.
    * `+/- <mad>` at n>=3, with a relative percentage appended once it passes
      5% -- the point at which a comparison between two columns starts being
      about the machine rather than about the libraries.
    """
    if repeats <= 1:
        return "n=1"
    if repeats == 2:
        return f"± {duration((high - low) / 2)}"
    text = f"± {duration(mad)}"
    median_ish = (low + high) / 2
    if median_ish > 0 and mad / median_ish > 0.05:
        text += f" ({mad / median_ish:.0%})"
    return text


def megabytes(value: float) -> str:
    """A MiB figure, or `n/a` for the NaN that means "not accountable here"."""
    return "n/a" if value != value else f"{value:.1f} MB"


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
            print(
                f"  {'case':<{width}} {'median':>11} {'spread':>13} "
                f"{'peak RAM':>11}  note"
            )
            print(f"  {'-' * width} {'-' * 11} {'-' * 13} {'-' * 11}  ----")
        if r.status != OK:
            print(
                f"  {r.case.label:<{width}} {r.status:>11} {'':>13} {'':>11}  {r.note}"
            )
            continue
        print(
            f"  {r.case.label:<{width}} {duration(r.seconds):>11} "
            f"{spread(r.mad, r.low, r.high, r.repeats):>13} "
            f"{megabytes(r.peak_mb):>11}  {r.note}"
        )
    print()


def _float(row: dict[str, str], field: str) -> float:
    """One CSV cell as a float, NaN when it is blank."""
    try:
        return float(row.get(field) or "nan")
    except ValueError:
        return float("nan")


#: The columns of the comparison table, as `(heading, width, how to render it)`.
#:
#: A list rather than an f-string per row so that the header, the rule and the
#: cells cannot drift apart -- which they had, leaving a `vs <base>` heading
#: over a ratio that was not against the base.
_COLUMNS: tuple[tuple[str, int, Any], ...] = (
    ("median", 11, lambda r: duration(_float(r, "seconds"))),
    (
        # Wide enough for the worst honest cell, `± 1000.0 ms (27%)`. The
        # percentage only appears on the rows that need it, and those are the
        # rows nobody should have to count characters to read.
        "spread",
        17,
        lambda r: spread(
            _float(r, "seconds_mad"),
            _float(r, "seconds_min"),
            _float(r, "seconds_max"),
            int(r.get("repeats") or 0),
        ),
    ),
    ("peak RAM", 10, lambda r: megabytes(_float(r, "peak_mb"))),
    ("peak RSS", 10, lambda r: megabytes(_float(r, "proc_peak_mb"))),
    ("store", 10, lambda r: _size(r.get("bytes"))),
)

#: Appended only when some row's CPU time diverges from its wall-clock, which is
#: what "this one used threads" looks like. Always-on it would be a column of
#: ratios near 1.0 that nobody reads; never it would leave the most interesting
#: difference between these writers invisible.
_CPU = ("cpu/wall", 9, lambda r: _ratio(_float(r, "cpu_seconds"), _float(r, "seconds")))

#: Suite-specific trailing columns, printed when the rows carry them. Named
#: rather than detected by position so a schema change cannot silently reorder
#: the table.
_EXTRA = {
    "levels": ("lvl", 5),
    "pyramid": ("pyramid", 10),
    "codec": ("codec", 14),
    "checksum": ("checksum", 13),
}


def _fit(value: str, width: int) -> str:
    """Pad or truncate to `width`. The CSV always keeps the full string.

    Truncated rather than allowed to push the rest of the row sideways: the
    `pyramid` cell spells out the whole divergence when a writer produced
    something else, which is exactly right in a file and ruins a table.
    """
    if len(value) > width:
        return value[: width - 1] + "…"
    return f"{value:<{width}}"


def _size(value: str | None) -> str:
    """A byte count as MiB, or blank."""
    if not value:
        return ""
    return f"{int(value) / (1024 * 1024):.1f} MiB"


def _ratio(numerator: float, denominator: float) -> str:
    """`numerator/denominator` as `1.83x`, or blank when either is missing."""
    if numerator != numerator or denominator != denominator or not denominator:
        return ""
    return f"{numerator / denominator:.2f}x"


def report_matrix(rows: list[dict[str, str]], schema: Schema) -> None:
    """Print one table per operation: a row per implementation and variant.

    Long rather than wide. The table used to put one column per implementation
    and one row per case, which reads beautifully right up to the point where
    the implementations stop having the same cases -- and with `method` and
    `[options.<impl>]` they do not. ngio's `mode=numpy` has no counterpart
    column in ngff-zarr, so a wide table spends most of its width on `—`.

    Grouped by operation, then by image, so everything inside one block is
    directly comparable and the ratio at the end means something. That ratio is
    per row against the first row that produced a number: an implementation that
    failed to install still has rows recording the failure, and anchoring on one
    of those would divide by an environment that never ran.
    """
    key, group = schema.column, schema.group
    blocks: list[str] = []
    for row in rows:
        if row[group] not in blocks and row[group] != "-":
            blocks.append(row[group])
    if not blocks:
        print(f"\nno results: every {key} failed to run\n")
        return

    for block in blocks:
        here = [r for r in rows if r[group] == block]
        for image in _distinct(here, "image"):
            _table(
                [r for r in here if r.get("image", "") == image], schema, block, image
            )
    _report_baselines(rows, key)
    print()


def _distinct(rows: list[dict[str, str]], field: str) -> list[str]:
    """The values of `field`, in the order they first appear."""
    seen: list[str] = []
    for row in rows:
        value = row.get(field, "")
        if value not in seen:
            seen.append(value)
    return seen


def _table(rows: list[dict[str, str]], schema: Schema, block: str, image: str) -> None:
    """One table: rows are `(impl, variant)`, columns are measurements."""
    key = schema.column
    columns = list(_COLUMNS)
    if any(
        _ratio(_float(r, "cpu_seconds"), _float(r, "seconds")) not in ("", "1.00x")
        and abs(_float(r, "cpu_seconds") - _float(r, "seconds"))
        > 0.2 * _float(r, "seconds")
        for r in rows
    ):
        columns.append(_CPU)
    trailing = [
        (field, *_EXTRA[field])
        for field in _EXTRA
        if any(r.get(field) for r in rows) and field in schema.fields
    ]

    names = _width(rows, key, len(key))
    # Dropped when nothing varies -- every row of `compare-io` -- rather than
    # printed as a column of blanks under a heading promising a distinction.
    variants = (
        _width(rows, "variant", len("variant"))
        if any(r.get("variant") for r in rows)
        else 0
    )
    # The first row with a number anchors the ratio. Rows are already in the
    # order the config named the implementations, so this is the first one the
    # config asked for that actually ran -- not whichever finished first.
    base = next((_float(r, "seconds") for r in rows if r.get("status") == OK), None)

    title = f"\n{block}"
    if image:
        title += f"   image={image}"
    repeats = {r.get("repeats") for r in rows if r.get("repeats")}
    if len(repeats) == 1:
        title += f"   repeats={repeats.pop()}"
    print(title)

    head = f"  {key:<{names}}" + (f"  {'variant':<{variants}}" if variants else "")
    rule = f"  {'-' * names}" + (f"  {'-' * variants}" if variants else "")
    for heading, width, _ in columns:
        head += f" {heading:>{width}}"
        rule += f" {'-' * width}"
    for _, heading, width in trailing:
        head += f"  {heading:<{width}}"
        rule += f"  {'-' * width}"
    if base:
        head += f" {'vs first':>9}"
        rule += f" {'-' * 9}"
    print(head)
    print(rule)

    for row in rows:
        line = f"  {row[key]:<{names}}" + (
            f"  {row.get('variant', ''):<{variants}}" if variants else ""
        )
        if row.get("status", OK) != OK or not row.get("seconds"):
            # A status, never a blank. The four of them are different claims --
            # see `core.measure` -- and an unlabelled gap in a comparison reads
            # as a result.
            span = sum(width for _, width, _ in columns) + len(columns) - 1
            status = row.get("status", "failed")
            print(f"{line} {status:>{span}}  {row.get('note', '')}")
            continue
        for _, width, render in columns:
            line += f" {render(row):>{width}}"
        for field, _, width in trailing:
            line += f"  {_fit(row.get(field, ''), width)}"
        if base:
            line += f" {_ratio(_float(row, 'seconds'), base):>9}"
        print(line)


def _width(rows: list[dict[str, str]], field: str, floor: int) -> int:
    """Column width for `field`, never narrower than its heading."""
    return max([floor, *(len(r.get(field, "")) for r in rows)])


def _report_baselines(rows: list[dict[str, str]], key: str) -> None:
    """Print what each implementation cost before it did anything.

    `rss_base_mb` is the process high-water mark taken after this environment's
    imports and after the input was materialised, but before the first timed
    run. The `peak RSS` column above is the same mark taken at the end, so the
    difference is what the case itself cost and this is the price of admission.

    Worth separating because for several of these libraries the price of
    admission is the larger number, and a `peak RSS` column read without it
    ranks libraries by how much they import.
    """
    baselines: dict[str, list[float]] = {}
    for row in rows:
        value = _float(row, "rss_base_mb")
        if value == value:
            baselines.setdefault(row[key], []).append(value)
    if not baselines:
        return
    print("\nbaseline RSS before measuring (interpreter, imports, input array)")
    width = max(len(name) for name in baselines)
    for name, values in baselines.items():
        print(f"  {name:<{width}}  {min(values):>8.1f} MB")
