"""Turning CSV rows into the payload the page draws.

The shaping here is deliberately schema-driven rather than file-driven. A
`compare-create` file happens to sweep `image` and `method` and to leave seven
of its ten `AXIS_FIELDS` empty, but a config is free to sweep `max_threads` or
`writer` instead -- so the axes are *discovered* (`axes`), assigned to roles by
preference (`roles`), and offered to the page as pickers. Naming `image` and
`method` in the code would have produced a report for one file.

Nothing here names a suite either. What differs between the three -- which
column identifies a series, which audit column decides whether two bars are
comparable, what the page calls things -- arrives as a `Profile`, and this
module is the same code for all of them. See `_profile`.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, NamedTuple

from ngio_benchmarks.core.measure import OK
from ngio_benchmarks.core.output import check_csv, read_csv

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.report._profile import Profile

#: Columns holding a real number, and how many places the CSV wrote.
#: `loadavg` is NaN on Windows and `peak_mb` is blank for adapters whose
#: allocation happens in C++ or Rust buffers `tracemalloc` never sees -- both
#: arrive as an empty cell and must stay `None` rather than becoming `0.0`,
#: which would read as "measured, and it was nothing".
#:
#: Not every suite writes every one of these: `sync_seconds` and `rss_base_mb`
#: are the comparison suites' alone. A column a schema does not have is read as
#: blank, which is the same `None` and the same absent tooltip line.
FLOAT_FIELDS = (
    "seconds",
    "seconds_mad",
    "seconds_min",
    "seconds_max",
    "cpu_seconds",
    "peak_mb",
    "proc_peak_mb",
    "rss_base_mb",
    "loadavg",
    "sync_seconds",
)

#: Counts. `levels` is deliberately absent: the audit writes it as
#: `5 (metadata declares 3)` when a writer's metadata disagrees with what is on
#: disk, and that disagreement is the interesting part.
INT_FIELDS = ("repeats", "bytes")

#: What both runners write in the group and case columns of a row recording an
#: environment that never installed (`internal._cli._failure_row`,
#: `compare._suite._failure_row`). It is a placeholder, not a value: the row
#: must still appear -- an environment that failed is the report's business --
#: but `-` must never become an axis value anyone can facet by.
PLACEHOLDER = "-"

#: Below this the suite itself warns (`--list`), because a single sample of a
#: pyramid build on a laptop is not a measurement to compare against another.
MIN_USEFUL_REPEATS = 3


class Report(NamedTuple):
    """Everything the page needs, ready to be serialised."""

    rows: list[dict[str, Any]]
    axes: list[dict[str, Any]]
    roles: dict[str, str | None]
    #: The distinct values of the suite's identifying column -- the
    #: implementations, or the environments -- in the order the file introduces
    #: them. What the texture patterns and the legend are built from.
    columns: list[str]
    provenance: dict[str, Any]
    notices: list[str]
    profile: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        """The object embedded in the page as JSON."""
        return {
            "rows": self.rows,
            "axes": self.axes,
            "roles": self.roles,
            "columns": self.columns,
            "provenance": self.provenance,
            "notices": self.notices,
            "profile": self.profile,
        }


def _cell(row: dict[str, str], field: str) -> str:
    """A CSV cell, with the absent column and the placeholder both blank."""
    value = row.get(field, "")
    return "" if value == PLACEHOLDER else value


def _float(value: str) -> float | None:
    """A CSV cell as a float, or None for the blank that means "no number"."""
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return None if number != number else number


def _int(value: str) -> int | None:
    """A CSV cell as an int, or None."""
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def axes(rows: list[dict[str, str]], profile: Profile) -> list[dict[str, Any]]:
    """The axes this file actually varies along.

    An axis qualifies on two counts. It must have more than one distinct
    non-blank value -- a blank is "this option does not apply to this row"
    (`mode` is ngio's alone among the writers, and each internal block sweeps
    axes the other three leave empty), not a value to plot. And it must carry
    information the identifying column does not already carry: `impl_version`
    has one distinct value per implementation in a normal file, and folding that
    into the series label would name every bar twice. It becomes a real axis
    only when one implementation appears at two versions, which is the
    version-comparison the README describes.
    """
    found = []
    for field in profile.candidates:
        if field in profile.excluded:
            continue
        values = _first_seen(rows, field)
        if len(values) < 2:
            continue
        column = profile.schema.column
        if field != column and _determined_by_column(rows, field, column):
            continue
        found.append({"field": field, "values": values, "prefixed": _ambiguous(values)})
    return found


def _first_seen(rows: list[dict[str, str]], field: str) -> list[str]:
    """The field's values in the order the file introduces them.

    Not sorted. Rows are written in the order the cartesian product produced
    them, so first-appearance *is* the order the config asked for: `small`
    before `medium`, the implementations in the order `IMPLS` declares, and the
    environments in the order the config declares them. Sorting would put
    `medium` before `small` and bury ngio in the middle of its own comparison.
    Series colour keys on the name, never on this position, so an odd order can
    never repaint anything.
    """
    seen: list[str] = []
    for row in rows:
        value = _cell(row, field)
        if value and value not in seen:
            seen.append(value)
    return seen


def _ambiguous(values: list[str]) -> bool:
    """Whether these values need their field name to mean anything.

    `dask` and `zarrs-python` say what they are. `true` and `8` do not -- a
    series labelled "ngff-zarr - true" tells the reader nothing, and neither
    does one labelled "current - 1000", so an axis whose values are booleans or
    bare numbers is spelled `field=value`.
    """
    return any(
        v in ("true", "false") or v.replace(".", "", 1).isdigit() for v in values
    )


def _determined_by_column(rows: list[dict[str, str]], field: str, column: str) -> bool:
    """Whether `field` never varies within a single value of `column`."""
    seen: dict[str, set[str]] = {}
    for row in rows:
        if _cell(row, field):
            seen.setdefault(row[column], set()).add(row[field])
    return all(len(values) < 2 for values in seen.values())


def roles(found: list[dict[str, Any]], profile: Profile) -> dict[str, str | None]:
    """Assign the facet and group roles; everything left identifies a series.

    Only a preferred axis takes a role. An unclaimed role stays empty rather
    than conscripting whatever else varies, because the series is the better
    home for an unexpected axis: a file sweeping `writer` reads as one chart
    with three writers side by side, not as three charts with one bar each. The
    `internal` profile names no group preference at all for the same reason --
    its four blocks sweep disjoint axes, so any global group axis would caption
    three cards in four with a value they do not have.

    The identifying column is never auto-assigned away from the series -- it is
    what the suite compares, so it stays the thing bars are coloured by unless
    the reader moves it. All of this is a default; the pickers on the page
    reassign any discovered axis to any role.
    """
    column = profile.schema.column
    available = [axis["field"] for axis in found if axis["field"] != column]
    facet = next((f for f in profile.facet_preference if f in available), None)
    group = next(
        (f for f in profile.group_preference if f in available and f != facet), None
    )
    return {"facet": facet, "group": group}


def _dedupe(
    rows: list[dict[str, str]], profile: Profile
) -> tuple[list[dict[str, str]], int]:
    """Collapse re-runs of the same case, keeping the last.

    Runs append, so one file can hold the same case twice. The key includes the
    version column so re-running against a newer library keeps both rows -- that
    is a comparison, not a duplicate.

    It also includes the group column, and that is not belt and braces. A
    `compare-io` case label is built from the axis labels alone
    (`core.axes.Case.label`), and the only axis there is the image -- so all six
    operations of one implementation carry the case `image=small`, and a key
    without the operation would silently collapse a 36-row file to six, keeping
    whichever operation ran last.
    """
    kept: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row[profile.schema.column],
            row.get(profile.version_field, ""),
            row.get(profile.schema.group, ""),
            row["case"],
        )
        kept[key] = row
    return list(kept.values()), len(rows) - len(kept)


def colours(rows: list[dict[str, str]], profile: Profile) -> dict[str, int]:
    """Which palette slot each series takes, over the whole file.

    By name against the profile's pinned order, never by rank and never by
    position among the rows that survived a filter. So ngio is the brand teal in
    every comparison file, a config that selects two writers gets the same two
    hues it would have got selecting six, and hiding a series on the page can
    never repaint the ones left -- which is the promise the stylesheet's header
    makes and the reason colour is safe to read at all.

    A name the profile does not pin -- every `internal` environment, since a
    config names those itself -- takes the lowest free slot in the order the
    file introduces it. Past the last slot the CSS variable falls back to grey,
    and the label and the texture carry identity instead.
    """
    names = _first_seen(rows, profile.schema.column)
    slots = {
        name: profile.palette.index(name) for name in names if name in profile.palette
    }
    spoken = set(slots.values())
    free = (i for i in range(len(names) + len(profile.palette)) if i not in spoken)
    return {name: slots[name] if name in slots else next(free) for name in names}


def _comparison(row: dict[str, str], profile: Profile) -> tuple[str, ...]:
    """The identity of a comparison cell: everything but who ran it.

    The group and every axis the schema declares, so two rows share a cell
    exactly when they were asked to do the same thing to the same input. The
    identifying column is excluded, because comparing implementations *within* a
    cell is the whole point.
    """
    return (
        _cell(row, profile.schema.group),
        *(_cell(row, field) for field in profile.schema.axis_fields),
    )


def _consensus(rows: list[dict[str, str]], profile: Profile) -> dict[tuple, str]:
    """What most of each comparison cell agreed the bytes were.

    Every read in one cell opens the same store (`compare.io._fixture` writes it
    once, with plain zarr-python, so it was not produced by a contestant), and a
    digest that disagrees is a row that read something else. Majority rather
    than "matches ngio", because the interesting failure is ngio being the odd
    one out and a rule anchored on ngio could not report that. A tie breaks
    toward the first row of the cell, which is the order `IMPLS` declares.
    """
    assert profile.caveat is not None
    cells: dict[tuple, list[str]] = {}
    for row in rows:
        digest = _cell(row, profile.caveat.field)
        if digest and row["status"] == OK:
            cells.setdefault(_comparison(row, profile), []).append(digest)
    agreed = {}
    for key, digests in cells.items():
        counts = Counter(digests)
        best = max(counts.values())
        agreed[key] = next(d for d in digests if counts[d] == best)
    return agreed


def _fair(
    row: dict[str, str], profile: Profile, consensus: dict[tuple, str]
) -> bool | None:
    """Whether this row's timing is comparable with the ones beside it.

    `None` rather than `False` when there is nothing to check: a suite with no
    audit column at all, or a row whose audit cell is blank because the writer
    never got far enough to be audited, or -- in `compare-io` -- a write, which
    carries no checksum by design. "We did not check" and "it differs" are
    different claims and the page draws them differently.
    """
    caveat = profile.caveat
    if caveat is None:
        return None
    value = _cell(row, caveat.field)
    if not value:
        return None
    if caveat.expected:
        return value == caveat.expected
    return value == consensus.get(_comparison(row, profile), value)


def _record(
    row: dict[str, str],
    fields: list[str],
    profile: Profile,
    consensus: dict[tuple, str],
) -> dict[str, Any]:
    """One CSV row as the page sees it.

    The key set is a fixed superset across all three suites -- a column a schema
    does not have arrives blank, exactly as an unfilled column of a schema that
    does have it would. So the page's record shape never varies by suite, and
    which of these keys a reader actually sees is the profile's decision rather
    than a shape the JavaScript has to branch on.
    """
    numbers = {name: _float(row.get(name, "")) for name in FLOAT_FIELDS}
    counts = {name: _int(row.get(name, "")) for name in INT_FIELDS}
    seconds, cpu = numbers["seconds"], numbers["cpu_seconds"]
    proc_peak, rss_base = numbers["proc_peak_mb"], numbers["rss_base_mb"]
    return {
        "column": row[profile.schema.column],
        "axes": {field: _cell(row, field) for field in fields},
        "status": row["status"],
        "note": row["note"],
        "case": row["case"],
        "variant": row.get("variant", ""),
        "methodNative": row.get("method_native", ""),
        "seconds": seconds,
        "mad": numbers["seconds_mad"],
        "low": numbers["seconds_min"],
        "high": numbers["seconds_max"],
        "repeats": counts["repeats"],
        "cpuSeconds": cpu,
        # cpu/wall above 1 means the library used threads. The ASCII table only
        # prints it when a row diverges by more than 20%; here it gets a panel,
        # so it is computed for every row that has both numbers.
        "parallelism": (cpu / seconds) if seconds and cpu is not None else None,
        "peakMb": numbers["peak_mb"],
        "procPeakMb": proc_peak,
        "rssBaseMb": rss_base,
        # The split the schema comment insists on: what the case cost, against
        # what merely importing the library cost. Clamped at zero because the
        # high-water mark is process-wide and a case that allocated nothing can
        # measure a hair under its own baseline. `None` for `internal`, which
        # takes no baseline -- its blocks share one interpreter, so there is no
        # per-case import to subtract, and its memory view reads `peakMb`.
        "caseMb": (
            max(proc_peak - rss_base, 0.0)
            if proc_peak is not None and rss_base is not None
            else None
        ),
        "syncSeconds": numbers["sync_seconds"],
        "bytes": counts["bytes"],
        "checksum": row.get("checksum", ""),
        "levels": row.get("levels", ""),
        "levelShapes": row.get("level_shapes", ""),
        "pyramid": row.get("pyramid", ""),
        "codec": row.get("codec", ""),
        "chunks": row.get("chunks", ""),
        "shards": row.get("shards", ""),
        "zarrFormat": row.get("zarr_format", ""),
        "version": row.get(profile.version_field, ""),
        "python": row["python"],
        "zarr": row["zarr"],
        "platform": row["platform"],
        "fair": _fair(row, profile, consensus),
    }


def _provenance(
    rows: list[dict[str, str]], records: list[dict[str, Any]], profile: Profile
) -> dict:
    """The strip above the charts: where these numbers came from.

    Every key is emitted for every suite, including the ones a schema has no
    column for. The masthead reads them positionally before it filters the empty
    ones out, so a missing key would throw inside the single `render()` call and
    leave the whole page blank.
    """

    def distinct(field: str) -> list[str]:
        return sorted({v for row in rows if (v := _cell(row, field))})

    repeats = [r["repeats"] for r in records if r["repeats"]]
    statuses = {
        status: sum(1 for r in records if r["status"] == status)
        for status in sorted({r["status"] for r in records})
    }
    versions: dict[str, set[str]] = {}
    for row in rows:
        if version := row.get(profile.version_field, ""):
            versions.setdefault(row[profile.schema.column], set()).add(version)
    return {
        "groups": distinct(profile.schema.group),
        "python": distinct("python"),
        "zarr": distinct("zarr"),
        "platform": distinct("platform"),
        "zarrFormat": distinct("zarr_format"),
        "versions": {name: sorted(v) for name, v in sorted(versions.items())},
        "statuses": statuses,
        "rows": len(records),
        "minRepeats": min(repeats) if repeats else None,
        "maxRepeats": max(repeats) if repeats else None,
    }


def _payload(profile: Profile, slots: dict[str, int]) -> dict[str, Any]:
    """The profile as the page receives it.

    Key by key rather than `_asdict()`, so a field that exists for Python's
    benefit never leaks into the page, and so the names the JavaScript reads are
    visible here rather than inferred from a NamedTuple's field order.
    """
    caveat = profile.caveat
    return {
        "suite": profile.suite,
        "column": profile.schema.column,
        "group": profile.schema.group,
        "columnLabel": profile.column_label,
        "groupLabel": profile.group_label,
        "versionsLabel": profile.versions_label,
        "eyebrow": profile.eyebrow,
        "title": profile.title,
        "coverageSubhead": profile.coverage_subhead,
        "memorySubhead": profile.memory_subhead,
        "memory": profile.memory,
        "baseline": profile.baseline,
        "colours": slots,
        "details": [list(detail) for detail in profile.details],
        "tables": {
            "timing": [list(column) for column in profile.timing_columns],
            "coverage": [list(column) for column in profile.coverage_columns],
        },
        "coverageChip": profile.coverage_chip,
        "caveat": (
            None
            if caveat is None
            else {
                "mark": caveat.mark,
                "chip": caveat.chip,
                "legend": caveat.legend,
                "help": caveat.help,
            }
        ),
        # Assembled here so the page never conditionally concatenates prose.
        "legend": [*profile.legend, *([] if caveat is None else [caveat.help])],
    }


def shape(path: Path, profile: Profile) -> Report:
    """Read a results CSV and shape it for the page.

    Raises `SystemExit` through `check_csv` when the header is not this suite's,
    which is the same refusal the runner gives when asked to append to a file
    written with different columns -- and the reason each suite keeps its own
    report command rather than one that sniffs.
    """
    if not check_csv(path, profile.schema):
        raise SystemExit(f"{path} is empty -- run the suite before reporting on it.")
    raw = read_csv(path)
    if not raw:
        raise SystemExit(f"{path} has a header but no rows.")

    rows, duplicates = _dedupe(raw, profile)
    found = axes(rows, profile)
    fields = [axis["field"] for axis in found]
    consensus = (
        _consensus(rows, profile)
        if profile.caveat is not None and not profile.caveat.expected
        else {}
    )
    records = [_record(row, fields, profile, consensus) for row in rows]
    assigned = roles(found, profile)
    provenance = _provenance(rows, records, profile)
    provenance["duplicates"] = duplicates

    notices = []
    if duplicates:
        notices.append(
            f"{duplicates} duplicate row{'s' if duplicates > 1 else ''} collapsed — "
            f"the same {profile.column_label}, version, {profile.group_label} and "
            "case appeared more than once, and the last of each was kept."
        )
    low = provenance["minRepeats"]
    if low and low < MIN_USEFUL_REPEATS:
        how_often = "once" if low == 1 else f"{low} times"
        every = low == provenance["maxRepeats"]
        notices.append(
            f"{'Every case was' if every else 'Some cases were'} timed {how_often}. "
            "There is no run-to-run spread to report from that, so those bars carry "
            "no whisker rather than a zero-width one that would claim a precision "
            f"nobody measured. Raise `repeats` to {MIN_USEFUL_REPEATS} or more "
            "before reading a difference between two of these numbers as real."
        )
    if profile.mixed_group_notice and len(provenance["groups"]) > 1:
        notices.append(
            profile.mixed_group_notice.format(values=", ".join(provenance["groups"]))
        )
    if profile.caveat is not None:
        differing = sum(1 for r in records if r["fair"] is False)
        if differing:
            notices.append(
                profile.caveat.notice.format(
                    n=differing, s="s" if differing > 1 else ""
                )
            )

    slots = colours(rows, profile)
    return Report(
        records,
        found,
        assigned,
        list(slots),
        provenance,
        notices,
        _payload(profile, slots),
    )
