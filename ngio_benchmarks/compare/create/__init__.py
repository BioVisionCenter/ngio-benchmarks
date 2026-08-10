"""Building an OME-Zarr compared across the libraries that write one.

One operation: take an array in memory, write a multiscale pyramid to disk with
valid OME-NGFF metadata. Every library here does that; almost none of them does
it the same way.

That is why this suite records more than a duration. The peers disagree on the
downsampling filter (nearest, mean, gaussian, ITK bin-shrink), on default
chunking, on which metadata they write, and on whether they hold the whole
pyramid in memory. A writer that is twice as fast because it used a nearest
downsample and skipped a level has not written the same artefact, so every row
also carries:

* `bytes` -- total size of the store it produced
* `levels` -- how many resolution levels actually exist afterwards
* `downsample` -- the filter it used, as the library names it

Read those three before reading the timing. Where a filter can be chosen, each
adapter picks the closest available to a plain box mean, so the comparison is
between implementations rather than between filters -- but `downsample` records
what was really used, because "closest available" is not the same thing.
"""

from __future__ import annotations

from ngio_benchmarks.compare.create._ops import audit
from ngio_benchmarks.core.output import Schema

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

SCHEMA = Schema(
    fields=(
        "impl",
        "impl_version",
        "python",
        "zarr",
        "platform",
        "op",
        "case",
        "image",
        "zarr_format",
        "levels",
        "level_shapes",
        "pyramid",
        "downsample",
        "seconds",
        "peak_mb",
        "proc_peak_mb",
        "bytes",
        "status",
        "note",
    ),
    axis_fields=("image",),
    column="impl",
    group="op",
)
