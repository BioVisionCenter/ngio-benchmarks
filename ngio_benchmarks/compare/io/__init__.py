"""Array access compared across libraries: what does ngio's layer cost?

Six operations, flat rather than a `direction x shape` product, because the
product would be rectangular and the matrix is not -- six labels need no skip
machinery, and every one of them is a sentence someone actually says:

| operation               | what it does                                     |
| ----------------------- | ------------------------------------------------ |
| `read_full`             | read level 0 back as an array                     |
| `read_roi_aligned`      | read a region whose bounds fall on chunk edges    |
| `read_roi_straddling`   | read the same-sized region offset into the chunks |
| `write_full`            | write the whole array into a fresh store          |
| `write_roi_aligned`     | write one aligned region                          |
| `write_roi_straddling`  | write one straddling region                       |

The aligned/straddling pair is the read amplification question the `internal`
`roi` block asks of ngio alone, asked here of everyone: a read inside chunk
boundaries touches the minimum number of chunks, and one shifted by a few
pixels fetches and discards every chunk on each edge. Libraries differ in how
much of that they avoid, and the ratio between the two columns is the answer.

Read operations all open **the same store**, written once by the parent with
plain zarr-python (see `_fixture`). Peers open the level-0 array directly and
ignore the OME metadata; ngio opens the container properly. Same bytes, same
layout, so the comparison is honestly "ngio's read path versus a raw array read
of it" rather than a comparison of two different files.
"""

from __future__ import annotations

from ngio_benchmarks.core.output import Schema

OPS = (
    "read_full",
    "read_roi_aligned",
    "read_roi_straddling",
    "write_full",
    "write_roi_aligned",
    "write_roi_straddling",
)

IMPLS = {
    "ngio": "ngio_benchmarks.compare.io.adapters.ngio_impl",
    "zarr": "ngio_benchmarks.compare.io.adapters.zarr_python",
    "zarrs": "ngio_benchmarks.compare.io.adapters.zarrs",
    "dask": "ngio_benchmarks.compare.io.adapters.dask_array",
    "tensorstore": "ngio_benchmarks.compare.io.adapters.tensorstore_impl",
    "z5py": "ngio_benchmarks.compare.io.adapters.z5py_impl",
    "acquire-zarr": "ngio_benchmarks.compare.io.adapters.acquire",
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
        "seconds",
        "peak_mb",
        "proc_peak_mb",
        "bytes",
        "checksum",
        "status",
        "note",
    ),
    axis_fields=("image",),
    column="impl",
    group="op",
)
