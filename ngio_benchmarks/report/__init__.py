"""Turning a results CSV into one interactive HTML page.

The ASCII table in `core.output` is the right shape for a handful of rows read
in the terminal straight after a run. `reference-compare-create.csv` is 88 rows
across six writers, two images and four downsampling filters, and a third of
them are `unsupported`; a `compare-io` sweep is six operations deep across seven
libraries. Both are past the size where scrolling a table is how anyone reads it.

The page is a single file with no network references, so it opens from
`file://` on a machine that has never seen this repository.

## What this module is *not*

It is not a plotting library and it pulls in no dependencies. `pyproject.toml`
keeps `dependencies` to numpy and zarr because everything importable by
`_child.py` has to install inside a peer library's environment; a report that
required pandas and plotly would either break that or need an extra nobody
remembers to install. A hundred rows do not need a dataframe, so the charts are
hand-rolled SVG with the data inlined as JSON.

Nothing here is imported from `compare.create.__init__` or `internal._run`,
which a child *does* import -- the dependency runs one way only.

## Three suites, one engine

One command per suite, each scoped to one schema: `check_csv` rejects a foreign
file at the door rather than half-rendering it. What the three share is
everything below that -- the same three views, the same formatters, the same
axis discovery, the same stylesheet -- so the pages read as one family and a
fix to any of it reaches all three.

What differs is a `Profile` (see `_profile`): which column identifies a series,
which audit column decides whether two bars measured the same artefact
(`pyramid` for the writers, `checksum` for the readers, nothing for `internal`),
which memory figure the suite actually recorded, and what the page calls things.

Within a schema each report targets the *schema*, not any one file: the axes are
discovered from the data (see `_model.axes`), so a config that sweeps
`max_threads` charts as readily as one that sweeps `method`.
"""

from __future__ import annotations

from ngio_benchmarks.report._model import Report, shape
from ngio_benchmarks.report._page import render
from ngio_benchmarks.report._profile import PROFILES, Profile

__all__ = ["PROFILES", "Profile", "Report", "render", "shape"]
