"""Building an OME-Zarr compared across the libraries that write one.

One operation: take an array in memory, write a multiscale pyramid to disk with
valid OME-NGFF metadata. Every library here does that; almost none of them does
it the same way.

That is why this suite records more than a duration. The peers disagree on the
downsampling filter (nearest, mean, gaussian, ITK bin-shrink), on default
chunking, on the codec, on which metadata they write, and on whether they hold
the whole pyramid in memory. A writer that is twice as fast because it used a
nearest downsample and skipped a level has not written the same artefact, so
every row also carries, all read back off disk after the fact:

* `bytes` -- total size of the store it produced
* `levels`, `level_shapes`, `pyramid` -- the pyramid that actually exists
* `codec`, `chunks`, `shards` -- how level 0 was actually stored

Read those before reading the timing.

## The `method` axis

The filter used to be whatever each adapter picked, recorded as a free-text
string. That made the table a comparison of filters wearing the libraries'
names: ngff-zarr's itkwasm gaussian against ome-zarr-py's anti-aliased resize
against bioio's order-0 decimation is not a comparison of writers.

`method` is now a real axis with a canonical vocabulary -- `default`, `mean`,
`nearest`, `linear`, `gaussian` -- that each adapter translates into its own
library's spelling and declares in its `METHODS` mapping. A value an adapter has
no translation for is `unsupported` for that row rather than silently served by
its default. `method_native` records what the library was actually handed.

`default` stays the default value, because "what does this library do when you
do not tell it anything" is the question this suite started from. `mean` and
`nearest` are the two values every writer here but one can honour, and are what
to sweep for a like-for-like comparison.

Anything with no cross-library equivalent -- ngio's consolidate mode,
ngff-zarr's tensorstore backend, bioio's three write methods, acquire-zarr's
thread count -- is declared per adapter in `OPTIONS` and swept from an
`[options.<impl>]` table.

## The `pipeline` axis

The other thing every writer here shares is that most of them encode through
zarr-python, and `zarrs` replaces zarr-python's codec pipeline for whoever is
holding it. So it is not a seventh writer -- it is a modifier on four of the six
-- and it is the suite's second canonical axis rather than anyone's option.

It earned that the hard way. It was visible only as `[options.iohub]
implementation`, where iohub's registry picks `zarrs-python` for itself, and
roughly half of iohub's headline advantage turned out to be a measurement of
zarrs inside a column labelled with iohub's name. The same question is worth
asking of the others, and `pipeline = ["zarr-python", "zarrs"]` asks it of all
of them at once. iohub keeps its own option and declines the axis: choosing its
zarr stack twice, in two vocabularies, is not a row anyone can read.

See `compare._pipeline`.
"""

from __future__ import annotations

from ngio_benchmarks.compare.create._ops import audit
from ngio_benchmarks.core.output import MEASUREMENT_FIELDS, Schema

__all__ = ["IMPLS", "OPS", "SCHEMA", "audit"]

#: One operation. Kept as an axis anyway so both comparison suites share a
#: runner and produce the same shape of table.
OPS = ("create_pyramid",)

IMPLS = {
    "ngio": "ngio_benchmarks.compare.create.adapters.ngio_impl",
    "ngff-zarr": "ngio_benchmarks.compare.create.adapters.ngff_zarr_impl",
    "ome-zarr-py": "ngio_benchmarks.compare.create.adapters.ome_zarr_py",
    "bioio": "ngio_benchmarks.compare.create.adapters.bioio",
    "iohub": "ngio_benchmarks.compare.create.adapters.iohub_impl",
    "acquire-zarr": "ngio_benchmarks.compare.create.adapters.acquire",
}

#: The axes a create case can vary along, as a hand-maintained literal.
#:
#: `method` and `pipeline` are the suite's own canonical axes -- the
#: downsampling filter and the codec pipeline, both of which more than one
#: library can express; the rest are individual adapters' `OPTIONS`, unioned.
#: Deliberately not computed by
#: importing the adapters -- same reason as `internal._run.AXIS_FIELDS`: an
#: adapter is allowed to fail to import inside a narrower environment, which
#: would give a child a shorter header than its parent and a merge that
#: `KeyError`s. A stale entry costs one empty column; `--list` makes a
#: mismatch obvious.
AXIS_FIELDS = (
    "image",
    "method",
    "method_native",
    "pipeline",
    "mode",
    "order",
    "use_tensorstore",
    "scale_strategy",
    "implementation",
    "writer",
    "max_threads",
)

SCHEMA = Schema(
    fields=(
        "impl",
        "impl_version",
        "python",
        "zarr",
        # Which zarrs encoded the row, blank on zarr-python's own pipeline.
        # Beside `zarr` rather than in `AXIS_FIELDS`, because it plays
        # `impl_version`'s part for the pipeline: provenance for a value on the
        # row, not something a config sweeps.
        "pipeline_version",
        "platform",
        "op",
        "case",
        *AXIS_FIELDS,
        "variant",
        "zarr_format",
        "levels",
        "level_shapes",
        "pyramid",
        "codec",
        "chunks",
        "shards",
        *MEASUREMENT_FIELDS,
        "sync_seconds",
        "bytes",
        "status",
        "note",
    ),
    axis_fields=AXIS_FIELDS,
    column="impl",
    group="op",
)
