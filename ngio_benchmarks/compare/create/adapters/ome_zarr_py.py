"""ome-zarr-py: the OME project's own writer.

`write_image` takes the array, the scale factors, the filter and the storage
options, and does the rest. The closest thing in this table to a one-liner, and
the natural baseline for "what does the obvious way cost".

`Scaler` is deprecated in 0.18 in favour of the `method` argument, so this uses
the latter. That matters for more than tidiness: passing `Scaler(method=
"gaussian")` is accepted and then silently downgraded to `RESIZE`, which would
put a row labelled gaussian in the table that was not one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.images import zarr_compressors
from ngio_benchmarks.core.measure import Measured, Unsupported

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

#: Canonical name -> the `Methods` member, by value rather than by attribute, so
#: this module still imports where `ome_zarr` is absent -- which is the whole
#: adapter protocol.
#:
#: `LOCAL_MEAN` is `skimage.downscale_local_mean`, a true box mean. `RESIZE` is
#: an anti-aliased order-1 resize, which is what `default` gets because it is
#: what the library does untold. There is no gaussian: 0.18 dropped it.
METHODS = {
    "default": "resize",
    "mean": "local_mean",
    "nearest": "nearest",
    "linear": "resize",
}

OPTIONS: dict[str, object] = {}


def build(
    op: str,
    spec: ImageSpec,
    root: Path,
    *,
    method: str = "default",
    method_native: str | None = None,
    **options: str,
) -> Measured:
    """Write the pyramid with `write_image`."""
    import zarr
    from ome_zarr.format import FormatV04, FormatV05
    from ome_zarr.scale import Methods
    from ome_zarr.writer import write_image

    chosen = method_native or METHODS[method]
    try:
        filter_ = Methods(chosen)
    except ValueError:
        raise Unsupported(
            f"ome-zarr-py has no {chosen!r} downsampling; it offers "
            f"{', '.join(m.value for m in Methods)}"
        ) from None

    data = _ops.source(spec, root)
    path = _ops.target(root, NAME, spec)
    y, x = spec.pixelsize
    fmt = FormatV04() if spec.zarr_format == 2 else FormatV05()

    storage: dict[str, object] = {"chunks": tuple(spec.chunks)}
    if spec.compressors != "auto":
        # `compressors` is the zarr v3 spelling and `compressor` the v2 one;
        # 0.18 forwards whichever it is given straight to `create_array`.
        key = "compressors" if spec.zarr_format == 3 else "compressor"
        storage[key] = zarr_compressors(spec)
    if spec.shards:
        # Passed rather than dropped. Without this the `sharded` image spec
        # produced five sharded stores and one unsharded one, and the unsharded
        # one looked quick.
        storage["shards"] = tuple(spec.shards)

    def create() -> None:
        group = zarr.open_group(store=path, mode="w", zarr_format=spec.zarr_format)
        write_image(
            image=data,
            group=group,
            scale_factors=_ops.scale_factors(spec),
            method=filter_,
            fmt=fmt,
            axes=list(spec.axes),
            scale={"y": y, "x": x},
            storage_options=storage,
        )

    return Measured(
        create,
        extra={"target": str(path), "method_native": filter_.value},
        setup=_ops.fresh(path),
    )
