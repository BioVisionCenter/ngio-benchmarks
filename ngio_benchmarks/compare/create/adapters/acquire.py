"""acquire-zarr: builds the pyramid while the data streams past.

The one writer here that never holds the image. It is designed to be fed frames
by a microscope and to write every level as it goes, so its pyramid is a
side-effect of acquisition rather than a pass over a finished array.

That makes it the most interesting row in this table and the least directly
comparable one. Its peak memory should be flat where everyone else's tracks the
data; its wall-clock is measured on an array that is already in RAM, which is
not the situation it was built for. Both facts are in the note.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.compare.io.adapters.acquire import dimensions
from ngio_benchmarks.core.measure import Measured

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
PYTHON = None
REPEATS = 3

#: The one writer whose pyramid geometry cannot be pinned. `ArraySettings`
#: exposes the filter (`downsampling_method`) and the depth (`max_levels`) but
#: not which axes are halved, and its streaming policy halves z and does not
#: halve xy at every level. So `spec.downsample` does not reach this column: the
#: audit reports `pyramid: differs` and prints the shapes it really wrote.
_NOTE = "streams frames; pyramid geometry is acquire-zarr's own, not spec.downsample"


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """Stream the volume through a multiscale-enabled writer."""
    import acquire_zarr as az

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
                    downsampling_method=az.DownsamplingMethod.MEAN,
                    # `max_levels` counts the levels *below* level 0, unlike
                    # everyone else's level count. Verified against the store:
                    # passing `spec.levels` writes one array too many, which
                    # the `levels` column caught.
                    max_levels=spec.levels - 1,
                )
            ],
        )
        stream = az.ZarrStream(settings)
        stream.append(data)
        stream.close()

    return Measured(
        create, _NOTE, extra={"target": str(path), "downsample": "mean (streaming)"}
    )
