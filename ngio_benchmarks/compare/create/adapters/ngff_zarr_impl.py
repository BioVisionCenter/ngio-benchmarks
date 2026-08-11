"""ngff-zarr: the reference implementation of the NGFF data model itself.

Three steps rather than one call -- describe the image, build the multiscales,
write them -- and all three are inside the timing, because together they are
what the other libraries do in a single call.

Two things about this library are configurable here and are worth being explicit
about, because both silently changed the number before they were:

* the **filter**. Its default goes through `itkwasm`, a WebAssembly build of
  ITK: better anti-aliasing, a much heavier dependency, and a runtime cost that
  dominates this column. `ITKWASM_BIN_SHRINK` is the box mean, and
  `DASK_IMAGE_NEAREST` is what the cheap writers here do.
* the **backend**. `use_tensorstore=True` writes through tensorstore instead of
  zarr-python, which is a different implementation of the same operation and
  belongs in the table as such rather than hidden inside one column.

`config.memory_target` is pinned rather than swept. Left alone it defaults to
half of whatever RAM psutil reports free and decides between writing the array
in one go and writing it slab by slab -- so the same config measures two
different code paths on two machines, or on one machine twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare import _pipeline
from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.measure import Measured, Unsupported

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "ngff-zarr"
DISTRIBUTION = "ngff-zarr"
#: `dask-image` and `tensorstore` are not ngff-zarr's dependencies; they are
#: what its `DASK_IMAGE_*` methods and its `use_tensorstore` backend import.
#: Without them this adapter would declare a `nearest` and a backend it cannot
#: run. An adapter's `METHODS` and `OPTIONS` are promises, and `REQUIRES` is
#: what makes them keepable.
#:
#: Installing them unconditionally does not perturb the rows that do not use
#: them: an unimported wheel on disk costs nothing at runtime, and which
#: versions resolved is on every row already.
REQUIRES = (
    "ngff-zarr>=0.41",
    "dask-image>=2024.5",
    "tensorstore>=0.1.85",
    "numpy>=2",
)
SUPPORTS = frozenset({"create_pyramid"})
#: `to_ngff_zarr` writes NGFF 0.4 or 0.5, which is zarr v2 or v3 respectively.
FORMATS = frozenset({2, 3})
PYTHON = None
REPEATS = 3

#: Canonical name -> `Methods` value. By value, so this module imports without
#: `ngff_zarr` present.
#:
#: `default` is `None`, which is the library choosing for itself -- currently
#: itkwasm gaussian. Spelling it as `None` rather than as the gaussian keeps the
#: `default` row meaning "untold" even if that default moves.
METHODS = {
    "default": None,
    "mean": "itkwasm_bin_shrink",
    "nearest": "dask_image_nearest",
    "gaussian": "itkwasm_gaussian",
}

OPTIONS = {
    #: Closed: these two are the write backends, and there is no third.
    "use_tensorstore": {"false": False, "true": True},
    #: `pad` downsamples each level from the one above; `exact` reuses the
    #: images `to_multiscales` already computed. Different work, same pyramid.
    "scale_strategy": ["pad", "exact"],
}

#: Bytes. Above this, ngff-zarr switches to writing slab by slab. Pinned high
#: enough that the images in this suite take the single-pass path on every
#: machine, so the column measures one code path rather than two.
_MEMORY_TARGET = 8 * 1024**3


def excludes_pipeline(pipeline: str, values: dict) -> str:
    """Why a codec pipeline cannot apply to this case, or an empty string.

    The tensorstore backend does not go through zarr-python at all, so a
    non-default pipeline under it would be installed, ignored, and reported as
    a row that measured something it did not. It is the same store, written by
    the same code, as the `zarr-python` row beside it.

    Answered here rather than as an `Unsupported` at build time so the pairing
    shows up in `--list` and no child process is started to discover it.
    """
    if pipeline != _pipeline.DEFAULT and values.get("use_tensorstore"):
        return (
            f"ngff-zarr's tensorstore backend does not encode through "
            f"zarr-python, so {pipeline} cannot reach it"
        )
    return ""


def build(
    op: str,
    spec: ImageSpec,
    root: Path,
    *,
    method: str = "default",
    method_native: str | None = None,
    use_tensorstore: bool = False,
    scale_strategy: str = "pad",
    **options: str,
) -> Measured:
    """Describe, downsample, write."""
    import ngff_zarr
    from ngff_zarr.methods import Methods

    chosen = method_native or METHODS[method]
    filter_ = None
    if chosen is not None:
        try:
            filter_ = Methods(chosen)
        except ValueError:
            raise Unsupported(
                f"ngff-zarr has no {chosen!r} downsampling; it offers "
                f"{', '.join(m.value for m in Methods)}"
            ) from None

    ngff_zarr.config.memory_target = _MEMORY_TARGET

    data = _ops.source(spec, root)
    path = _ops.target(root, NAME, spec)
    dims = list(spec.axes)
    y, x = spec.pixelsize
    # Every dim, not just the two with a real pixel size: ngff-zarr indexes the
    # scale mapping by dim name while building the coordinate transformations,
    # so a missing key is a KeyError rather than a default.
    scale = {name: {"y": y, "x": x}.get(name, 1.0) for name in dims}
    chunks = dict(zip(dims, spec.chunks, strict=True))
    shards = None
    if spec.shards:
        shards = {
            name: shard // chunk
            for name, shard, chunk in zip(dims, spec.shards, spec.chunks, strict=True)
        }

    written: dict[str, object] = {"overwrite": True, "chunks_per_shard": shards}
    if use_tensorstore:
        written["use_tensorstore"] = True
    if spec.compressors != "auto":
        from ngff_zarr.codecs import codec_from_name

        # `compressor` is the spelling `to_ngff_zarr` translates for both zarr
        # formats; the plural is v3-only and raises on 0.4.
        written["compressor"] = codec_from_name(
            "none" if spec.compressors == "none" else spec.compressors
        )

    def create() -> None:
        image = ngff_zarr.to_ngff_image(data, dims=dims, scale=scale)
        multiscales = ngff_zarr.to_multiscales(
            image,
            # `explicit_ones`: ngff-zarr indexes every spatial dim of the
            # factor mapping, so an axis left out is a KeyError rather than an
            # axis left alone.
            scale_factors=_ops.scale_factors(spec, explicit_ones=True),
            method=filter_,
            chunks=chunks,
        )
        ngff_zarr.to_ngff_zarr(
            str(path),
            multiscales,
            version="0.4" if spec.zarr_format == 2 else "0.5",
            scale_strategy=scale_strategy,
            **written,
        )

    backend = "tensorstore" if use_tensorstore else "zarr-python"
    return Measured(
        create,
        f"{backend} backend, memory_target pinned to {_MEMORY_TARGET // 1024**3} GiB",
        extra={
            "target": str(path),
            "method_native": filter_.value if filter_ else "library default",
        },
        setup=_ops.fresh(path),
    )
