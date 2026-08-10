"""ome-zarr-py: the OME project's own writer.

`write_image` takes the array, the scale factors, and the storage options, and
does the rest. The closest thing in this table to a one-liner, and the natural
baseline for "what does the obvious way cost".

`Scaler` is deprecated in 0.18 in favour of the `scale_factors` argument, so
this uses the latter; `method` stays at the library default (`RESIZE`) for the
same reason ngio's mode does -- the question is what each library does when not
told otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "ome-zarr-py"
DISTRIBUTION = "ome-zarr"
REQUIRES = ("ome-zarr>=0.18", "numpy>=2")
SUPPORTS = frozenset({"create_pyramid"})
FORMATS = frozenset({2, 3})
PYTHON = None
REPEATS = 3


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """Write the pyramid with `write_image`."""
    import zarr
    from ome_zarr.format import FormatV04, FormatV05
    from ome_zarr.writer import write_image

    data = _ops.source(spec, root)
    path = _ops.target(root, NAME, spec)
    y, x = spec.pixelsize
    fmt = FormatV04() if spec.zarr_format == 2 else FormatV05()

    def create() -> None:
        group = zarr.open_group(store=path, mode="w", zarr_format=spec.zarr_format)
        write_image(
            image=data,
            group=group,
            scale_factors=_ops.scale_factors(spec),
            fmt=fmt,
            axes=list(spec.axes),
            scale={"y": y, "x": x},
            storage_options={"chunks": tuple(spec.chunks)},
        )

    return Measured(
        create, extra={"target": str(path), "downsample": "resize (default)"}
    )
