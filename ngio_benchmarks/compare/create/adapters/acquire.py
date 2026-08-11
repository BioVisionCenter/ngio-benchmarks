"""acquire-zarr: builds the pyramid while the data streams past.

The one writer here that never holds the image. It is designed to be fed frames
by a microscope and to write every level as it goes, so its pyramid is a
side-effect of acquisition rather than a pass over a finished array.

That makes it the most interesting row in this table and the least directly
comparable one. Its peak memory should be flat where everyone else's tracks the
data; its wall-clock is measured on an array that is already in RAM, which is
not the situation it was built for. Both facts are in the note.

It is also the only library here with a first-class thread count, which is why
`max_threads` is an option: every other column's parallelism is whatever its
dask scheduler or codec pipeline decided, and pinning one of them is at least
one honest data point about what threads are worth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.compare.io.adapters.acquire import dimensions
from ngio_benchmarks.core.measure import Measured, Unsupported

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "acquire-zarr"
DISTRIBUTION = "acquire-zarr"
REQUIRES = ("acquire-zarr>=0.8", "numpy>=2")
SUPPORTS = frozenset({"create_pyramid"})
#: acquire-zarr 0.8 dropped zarr v2: its `ZarrVersion` enum has only `V3`.
FORMATS = frozenset({3})
#: A C++ streaming writer; its buffers are invisible to `tracemalloc`.
NATIVE = True
#: Out of the `pipeline` axis: zarr-python is nowhere in this writer's path, so
#: swapping its codec pipeline would produce a second identical row wearing a
#: label that meant nothing. The default entry is here because it is the name
#: for "nothing installed", not a claim that this writer uses zarr-python.
PIPELINES = frozenset({"zarr-python"})
PYTHON = None
REPEATS = 3

#: Canonical name -> `DownsamplingMethod` member name. `DECIMATE` takes the top
#: left of each 2x2 block, which is nearest. `MIN` and `MAX` have no canonical
#: name and are reachable through `[options.acquire-zarr] method_native`.
METHODS = {
    "default": "MEAN",
    "mean": "MEAN",
    "nearest": "DECIMATE",
}

OPTIONS = {
    #: Open: any positive int. The only direct thread dial in this table.
    "max_threads": [1, 4, 8],
}

#: The one writer whose pyramid geometry cannot be pinned. `ArraySettings`
#: exposes the filter (`downsampling_method`) and the depth (`max_levels`) but
#: not which axes are halved, and its streaming policy halves z and does not
#: halve xy at every level. So `spec.downsample` does not reach this column: the
#: audit reports `pyramid: differs` and prints the shapes it really wrote.
_NOTE = "streams frames; pyramid geometry is acquire-zarr's own, not spec.downsample"


def build(
    op: str,
    spec: ImageSpec,
    root: Path,
    *,
    method: str = "default",
    method_native: str | None = None,
    max_threads: int | None = None,
    **options: str,
) -> Measured:
    """Stream the volume through a multiscale-enabled writer."""
    import acquire_zarr as az

    chosen = (method_native or METHODS[method]).upper()
    if not hasattr(az.DownsamplingMethod, chosen):
        raise Unsupported(
            f"acquire-zarr has no {chosen!r} downsampling; it offers "
            f"{', '.join(az.DownsamplingMethod.__members__)}"
        )
    filter_ = getattr(az.DownsamplingMethod, chosen)

    data = _ops.source(spec, root)
    path = _ops.target(root, NAME, spec)

    compression = None
    if spec.compressors not in ("none", "auto"):
        compression = az.CompressionSettings(
            compressor=az.Compressor.BLOSC1,
            codec=az.CompressionCodec.BLOSC_LZ4
            if spec.compressors == "lz4"
            else az.CompressionCodec.BLOSC_ZSTD,
            level=1,
        )

    def create() -> None:
        settings = az.StreamSettings(
            store_path=str(path),
            version=az.ZarrVersion.V3,
            overwrite=True,
            arrays=[
                az.ArraySettings(
                    output_key="0",
                    dimensions=dimensions(spec, az),
                    data_type=getattr(az.DataType, spec.dtype.upper()),
                    compression=compression,
                    downsampling_method=filter_,
                    # `max_levels` counts the levels *below* level 0, unlike
                    # everyone else's level count. Verified against the store:
                    # passing `spec.levels` writes one array too many, which
                    # the `levels` column caught.
                    max_levels=spec.levels - 1,
                )
            ],
        )
        if max_threads is not None:
            settings.max_threads = int(max_threads)
        stream = az.ZarrStream(settings)
        stream.append(data)
        stream.close()

    return Measured(
        create,
        _NOTE,
        extra={"target": str(path), "method_native": chosen.lower()},
        setup=_ops.fresh(path),
    )
