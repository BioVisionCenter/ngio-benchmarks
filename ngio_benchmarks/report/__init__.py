"""Turning a `compare-create` results CSV into one interactive HTML page.

The ASCII table in `core.output` is the right shape for a handful of rows read
in the terminal straight after a run. `reference-compare-create.csv` is 88 rows
across six writers, two images and four downsampling filters, and a third of
them are `unsupported` -- past the size where scrolling a table is how anyone
reads it.

The page is a single file with no network references, so it opens from
`file://` on a machine that has never seen this repository.

## What this module is *not*

It is not a plotting library and it pulls in no dependencies. `pyproject.toml`
keeps `dependencies` to numpy and zarr because everything importable by
`_child.py` has to install inside a peer library's environment; a report that
required pandas and plotly would either break that or need an extra nobody
remembers to install. 88 rows do not need a dataframe, so the charts are
hand-rolled SVG with the data inlined as JSON.

Nothing here is imported from `compare.create.__init__`, which a child *does*
import -- the dependency runs one way only.

## Scoped to one schema

`compare-create` only. `check_csv` rejects an `internal` or `compare-io` file
at the door rather than half-rendering it, because the interesting parts of
this page -- the pyramid and codec caveats drawn on the bars -- are columns
those schemas do not have.

Within that schema it targets the *schema*, not any one file: the axes are
discovered from the data (see `_model.axes`), so a config that sweeps
`max_threads` charts as readily as one that sweeps `method`.
"""

from __future__ import annotations

from ngio_benchmarks.report._model import Report, shape
from ngio_benchmarks.report._page import render

__all__ = ["Report", "render", "shape"]
