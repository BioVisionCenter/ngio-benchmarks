"""`ngio-bench-report-<suite>` -- a results CSV in, one HTML file out.

The three suites take a config file, because a config file *is* the experiment.
These commands are the other direction: the experiment already ran, and what is
left is reading it. So they do not route through `core.cli.parse`, whose two
arguments are `config` and `--list`, neither of which means anything here.

One command per suite, named after the suite it reads, for the same reason
there is no shared `ngio-bench` dispatcher. The three write different columns,
and each command's `check_csv` refuses the other two at the door rather than
half-rendering one of them. What they share is everything below the schema: one
model, one stylesheet, one chart engine, and a `Profile` describing what each is
looking at.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from ngio_benchmarks.report import _profile
from ngio_benchmarks.report._model import shape
from ngio_benchmarks.report._page import write


def _main(profile: _profile.Profile, argv: list[str] | None) -> int:
    """Render one suite's results CSV to a self-contained HTML report."""
    parser = argparse.ArgumentParser(
        prog=f"ngio-bench-report-{profile.command}",
        description=(
            f"Render a {profile.suite} results CSV as one self-contained, "
            "interactive HTML file."
        ),
    )
    parser.add_argument("csv", type=Path, help=f"a {profile.suite} results CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="where to write the page (default: the CSV path with .html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_",
        help="open the page in the default browser once written",
    )
    args = parser.parse_args(argv)

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} does not exist.")

    report = shape(args.csv, profile)
    destination = args.output or args.csv.with_suffix(".html")
    written = write(report, args.csv.name, destination)

    counts = ", ".join(
        f"{count} {status}" for status, count in report.provenance["statuses"].items()
    )
    print(f"{written}  ({report.provenance['rows']} rows: {counts})")
    for notice in report.notices:
        print(f"  note: {notice}")
    if args.open_:
        webbrowser.open(written.resolve().as_uri())
    return 0


def internal(argv: list[str] | None = None) -> int:
    """Render an `internal` results CSV."""
    return _main(_profile.INTERNAL, argv)


def compare_io(argv: list[str] | None = None) -> int:
    """Render a `compare-io` results CSV."""
    return _main(_profile.IO, argv)


def compare_create(argv: list[str] | None = None) -> int:
    """Render a `compare-create` results CSV."""
    return _main(_profile.CREATE, argv)
