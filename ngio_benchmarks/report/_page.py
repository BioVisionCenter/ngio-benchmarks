"""Assembling the single HTML file.

Everything is inlined -- stylesheet, script, data -- so the result opens from
`file://` with no network at all. That is not tidiness: these CSVs are produced
on laptops and cluster nodes and read somewhere else entirely, and a page that
needs a CDN to draw a bar chart is a page that renders blank on the machine
that has the numbers.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ngio_benchmarks.report._model import Report

ASSETS = Path(__file__).parent / "assets"


def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def _embed(payload: dict) -> str:
    r"""Serialise the payload for a `<script>` body.

    `</script>` inside a string literal closes the element wherever it appears,
    and the `note` column carries free text an adapter wrote. `<\\/` is the
    standard escape and parses back to the same JSON string. The line and
    paragraph separators that would otherwise break a script body are already
    escaped by `json.dumps`, which defaults to ASCII-safe output.
    """
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def render(report: Report, source: str) -> str:
    """The whole page, as one string."""
    title = f"ngio benchmarks · {report.profile['suite']} -- {source}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{_asset("report.css")}
</style>
</head>
<body>
<div id="app" aria-busy="true">
  <noscript>
    <p class="notice">This report draws its charts in the browser, so it needs
    JavaScript. The numbers themselves are in the CSV beside it.</p>
  </noscript>
</div>
<script id="report-data" type="application/json">{_embed(report.payload())}</script>
<script>
{_asset("report.js")}
</script>
</body>
</html>
"""


def write(report: Report, source: str, destination: Path) -> Path:
    """Render and write, returning the path written."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(report, source), encoding="utf-8")
    return destination
