"""Turning CSV rows into the payload the page draws.

The shaping here is deliberately schema-driven rather than file-driven. The
reference CSV happens to sweep `image` and `method` and to leave seven of the
ten `AXIS_FIELDS` empty, but a config is free to sweep `max_threads` or
`writer` instead -- so the axes are *discovered* (`axes`), assigned to roles by
preference (`roles`), and offered to the page as pickers. Naming `image` and
`method` in the code would have produced a report for one file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from ngio_benchmarks.compare.create import SCHEMA
from ngio_benchmarks.core.output import check_csv, read_csv

if TYPE_CHECKING:
    from pathlib import Path

#: Columns holding a real number, and how many places the CSV wrote.
#: `loadavg` is NaN on Windows and `peak_mb` is blank for adapters whose
#: allocation happens in C++ or Rust buffers `tracemalloc` never sees -- both
#: arrive as an empty cell and must stay `None` rather than becoming `0.0`,
#: which would read as "measured, and it was nothing".
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

#: Axis candidates, in the order they are offered as roles. `impl` leads
#: because it is `SCHEMA.column` -- the thing a comparison suite compares.
#: `method_native` is excluded on purpose: it is a description of what `method`
#: became inside one library (`dask/linear`, `decimate`), so it is detail on a
#: row rather than an axis anyone sweeps.
CANDIDATE_AXES = ("impl", "impl_version", *SCHEMA.axis_fields)
EXCLUDED_AXES = ("method_native",)

#: Which axes claim the facet and group roles when they vary. A role whose
#: preference is not in this file is simply absent, and the axis it would have
#: held stays in the series -- see `roles`.
FACET_PREFERENCE = ("image",)
#: `pipeline` after `method` rather than instead of it: a file that swept only
#: the codec pipeline then draws zarr-python and zarrs side by side inside each
#: library, which is the whole shape of that question.
GROUP_PREFERENCE = ("method", "pipeline")

#: The pyramid audit's way of saying the writer built what it was asked for.
#: Anything else spells out the divergence, and a timing whose pyramid differs
#: is not comparable with one whose pyramid does not.
AS_ASKED = "as asked"

#: Below this the suite itself warns (`--list`), because a single sample of a
#: pyramid build on a laptop is not a measurement to compare against another.
MIN_USEFUL_REPEATS = 3


class Report(NamedTuple):
    """Everything the page needs, ready to be serialised."""

    rows: list[dict[str, Any]]
    axes: list[dict[str, Any]]
    roles: dict[str, str | None]
    impls: list[str]
    provenance: dict[str, Any]
    notices: list[str]

    def payload(self) -> dict[str, Any]:
        """The object embedded in the page as JSON."""
        return {
            "rows": self.rows,
            "axes": self.axes,
            "roles": self.roles,
            "impls": self.impls,
            "provenance": self.provenance,
            "notices": self.notices,
        }


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


def axes(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """The axes this file actually varies along.

    An axis qualifies on two counts. It must have more than one distinct
    non-blank value -- a blank is "this option does not apply to this
    implementation" (`mode` is ngio's alone), not a value to plot. And it must
    carry information `impl` does not already carry: `impl_version` has six
    distinct values in a normal file, one per implementation, and folding that
    into the series label would name every bar twice. It becomes a real axis
    only when one implementation appears at two versions, which is the
    version-comparison the README describes.
    """
    found = []
    for field in CANDIDATE_AXES:
        if field in EXCLUDED_AXES:
            continue
        values = _first_seen(rows, field)
        if len(values) < 2:
            continue
        if field != "impl" and _determined_by_impl(rows, field):
            continue
        found.append({"field": field, "values": values, "prefixed": _ambiguous(values)})
    return found


def _first_seen(rows: list[dict[str, str]], field: str) -> list[str]:
    """The field's values in the order the file introduces them.

    Not sorted. Rows are written in the order the cartesian product produced
    them, so first-appearance *is* the order the config asked for: `small`
    before `medium`, and the implementations in the order `IMPLS` declares.
    Sorting would put `medium` before `small` and bury ngio in the middle of
    its own comparison. Series colour keys on the implementation *name*, never
    on this position, so an odd order can never repaint anything.
    """
    seen: list[str] = []
    for row in rows:
        value = row.get(field)
        if value and value not in seen:
            seen.append(value)
    return seen


def _ambiguous(values: list[str]) -> bool:
    """Whether these values need their field name to mean anything.

    `dask` and `zarrs-python` say what they are. `true` and `8` do not -- a
    series labelled "ngff-zarr - true" tells the reader nothing, so an axis
    whose values are booleans or bare numbers is spelled `field=value`.
    """
    return any(
        v in ("true", "false") or v.replace(".", "", 1).isdigit() for v in values
    )


def _determined_by_impl(rows: list[dict[str, str]], field: str) -> bool:
    """Whether `field` never varies within a single implementation."""
    seen: dict[str, set[str]] = {}
    for row in rows:
        if row.get(field):
            seen.setdefault(row["impl"], set()).add(row[field])
    return all(len(values) < 2 for values in seen.values())


def roles(found: list[dict[str, Any]]) -> dict[str, str | None]:
    """Assign the facet and group roles; everything left identifies a series.

    Only a preferred axis takes a role. An unclaimed role stays empty rather
    than conscripting whatever else varies, because the series is the better
    home for an unexpected axis: a file sweeping `writer` reads as one chart
    with three writers side by side, not as three charts with one bar each.

    `impl` is never auto-assigned away from the series -- it is what the suite
    compares, so it stays the thing bars are coloured by unless the reader
    moves it themselves. All of this is a default; the pickers on the page
    reassign any discovered axis to any role.
    """
    available = [axis["field"] for axis in found if axis["field"] != "impl"]
    facet = next((f for f in FACET_PREFERENCE if f in available), None)
    group = next((f for f in GROUP_PREFERENCE if f in available and f != facet), None)
    return {"facet": facet, "group": group}


def _dedupe(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Collapse re-runs of the same case, keeping the last.

    Runs append, so one file can hold the same case twice. The key includes
    `impl_version` so re-running against a newer library keeps both rows --
    that is a comparison, not a duplicate.
    """
    kept: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        kept[(row["impl"], row["impl_version"], row["case"])] = row
    return list(kept.values()), len(rows) - len(kept)


def _record(row: dict[str, str], fields: list[str]) -> dict[str, Any]:
    """One CSV row as the page sees it."""
    numbers = {name: _float(row[name]) for name in FLOAT_FIELDS}
    counts = {name: _int(row[name]) for name in INT_FIELDS}
    seconds, cpu = numbers["seconds"], numbers["cpu_seconds"]
    proc_peak, rss_base = numbers["proc_peak_mb"], numbers["rss_base_mb"]
    return {
        "impl": row["impl"],
        "axes": {field: row.get(field, "") for field in fields},
        "status": row["status"],
        "note": row["note"],
        "case": row["case"],
        "variant": row["variant"],
        "methodNative": row["method_native"],
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
        # measure a hair under its own baseline.
        "caseMb": (
            max(proc_peak - rss_base, 0.0)
            if proc_peak is not None and rss_base is not None
            else None
        ),
        "syncSeconds": numbers["sync_seconds"],
        "bytes": counts["bytes"],
        "levels": row["levels"],
        "levelShapes": row["level_shapes"],
        "pyramid": row["pyramid"],
        "codec": row["codec"],
        "chunks": row["chunks"],
        "shards": row["shards"],
        "zarrFormat": row["zarr_format"],
        "implVersion": row["impl_version"],
        "python": row["python"],
        "zarr": row["zarr"],
        "platform": row["platform"],
        # None rather than False when the writer never got far enough to be
        # audited -- "we did not check" and "it differs" are different claims.
        "fair": (row["pyramid"] == AS_ASKED) if row["pyramid"] else None,
    }


def _provenance(rows: list[dict[str, str]], records: list[dict[str, Any]]) -> dict:
    """The strip above the charts: where these numbers came from."""

    def distinct(field: str) -> list[str]:
        return sorted({row[field] for row in rows if row.get(field)})

    repeats = [r["repeats"] for r in records if r["repeats"]]
    statuses = {
        status: sum(1 for r in records if r["status"] == status)
        for status in sorted({r["status"] for r in records})
    }
    versions = {}
    for row in rows:
        if row["impl_version"]:
            versions.setdefault(row["impl"], set()).add(row["impl_version"])
    return {
        "ops": distinct("op"),
        "python": distinct("python"),
        "zarr": distinct("zarr"),
        "platform": distinct("platform"),
        "zarrFormat": distinct("zarr_format"),
        "versions": {impl: sorted(v) for impl, v in sorted(versions.items())},
        "statuses": statuses,
        "rows": len(records),
        "minRepeats": min(repeats) if repeats else None,
        "maxRepeats": max(repeats) if repeats else None,
    }


def shape(path: Path) -> Report:
    """Read a `compare-create` CSV and shape it for the page.

    Raises `SystemExit` through `check_csv` when the header is not this
    suite's, which is the same refusal the runner gives when asked to append to
    a file written with different columns.
    """
    if not check_csv(path, SCHEMA):
        raise SystemExit(f"{path} is empty -- run the suite before reporting on it.")
    raw = read_csv(path)
    if not raw:
        raise SystemExit(f"{path} has a header but no rows.")

    rows, duplicates = _dedupe(raw)
    found = axes(rows)
    fields = [axis["field"] for axis in found]
    records = [_record(row, fields) for row in rows]
    assigned = roles(found)
    provenance = _provenance(rows, records)
    provenance["duplicates"] = duplicates

    notices = []
    if duplicates:
        notices.append(
            f"{duplicates} duplicate row{'s' if duplicates > 1 else ''} collapsed -- "
            "the same implementation, version and case appeared more than once, "
            "and the last of each was kept."
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
    if len(provenance["ops"]) > 1:
        notices.append(
            "This file mixes operations (" + ", ".join(provenance["ops"]) + "). "
            "The suite only defines create_pyramid, so the charts pool them."
        )
    differing = sum(1 for r in records if r["fair"] is False)
    if differing:
        notices.append(
            f"{differing} row{'s' if differing > 1 else ''} built a pyramid that "
            "differs from the one requested. Those bars are hatched -- they timed "
            "a different artefact, so they are not like-for-like."
        )

    impls = sorted({row["impl"] for row in rows})
    return Report(records, found, assigned, impls, provenance, notices)
