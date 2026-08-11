"""`ngio-bench-report-create` -- a results CSV in, one HTML file out.

The three suites take a config file, because a config file *is* the experiment.
This command is the other direction: the experiment already ran, and what is
left is reading it. So it does not route through `core.cli.parse`, whose two
arguments are `config` and `--list`, neither of which means anything here.

The name says `-create` rather than claiming a general `ngio-bench-report`,
because it understands one schema. `compare-io` and `internal` write different
columns and would need different charts; when they get one, they get their own
command, for the same reason there is no shared `ngio-bench` dispatcher.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from ngio_benchmarks.report._model import shape
from ngio_benchmarks.report._page import write


def main(argv: list[str] | None = None) -> int:
    """Render a `compare-create` CSV to a self-contained HTML report."""
    parser = argparse.ArgumentParser(
        prog="ngio-bench-report-create",
        description=(
            "Render a compare-create results CSV as one self-contained, "
            "interactive HTML file."
        ),
    )
    parser.add_argument("csv", type=Path, help="a compare-create results CSV")
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

    report = shape(args.csv)
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


if __name__ == "__main__":
    raise SystemExit(main())
