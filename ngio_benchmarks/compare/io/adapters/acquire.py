"""acquire-zarr: a streaming writer, and only a writer.

It has no read API at all, and its write is not the write the rest of this
table is doing: it appends frames to an open stream as an acquisition produces
them, rather than assigning into an existing array. So it declares exactly one
supported operation, and that one carries a permanent note saying what it
actually did.

That note is the point of including it. A number that is not comparable must
not sit in a comparison table unlabelled, and the honest way to handle an
implementation with a different model is to show it with its difference stated
rather than to leave it out and let the table imply nothing else exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "acquire-zarr"
DISTRIBUTION = "acquire-zarr"
REQUIRES = ("acquire-zarr>=0.8", "numpy>=2")
#: No read API, and a region write is not expressible in a frame stream.
SUPPORTS = frozenset({"write_full"})
#: acquire-zarr 0.8 dropped zarr v2: its `ZarrVersion` enum has only `V3`.
FORMATS = frozenset({3})
#: A C++ streaming writer; its buffers are invisible to `tracemalloc`.
NATIVE = True
PYTHON = None

_NOTE = "streams frames; not an array assignment"


def dimensions(spec: ImageSpec, settings_module):
    """Describe `spec`'s axes to acquire-zarr.

    The leading axis is the append dimension and is declared with size 0: that
    is how the library is told the stream grows there, and it is why this
    adapter can only express a whole-volume write.
    """
    az = settings_module
    kinds = {
        "t": az.DimensionType.TIME,
        "c": az.DimensionType.CHANNEL,
        "z": az.DimensionType.SPACE,
        "y": az.DimensionType.SPACE,
        "x": az.DimensionType.SPACE,
    }
    dims = []
    for position, name in enumerate(spec.axes):
        chunk = spec.chunks[position]
        shard = (spec.shards[position] // chunk) if spec.shards else 1
        dims.append(
            az.Dimension(
                name=name,
                kind=kinds[name],
                array_size_px=0 if position == 0 else spec.shape[position],
                chunk_size_px=chunk,
                shard_size_chunks=shard,
            )
        )
    return dims


def build(op: str, spec: ImageSpec, root: Path) -> Measured:
    """Stream the whole volume into a fresh store."""
    import acquire_zarr as az

    from ngio_benchmarks.compare.io import _ops

    path = _ops.target(root, NAME, op, spec)
    data = _ops.patch(spec, op, root)

    compression = None
    if spec.compressors not in ("none", "auto"):
        compression = az.CompressionSettings(
            compressor=az.Compressor.BLOSC1,
            codec=az.CompressionCodec.BLOSC_LZ4
            if spec.compressors == "lz4"
            else az.CompressionCodec.BLOSC_ZSTD,
            level=1,
        )

    def write() -> None:
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
                )
            ],
        )
        # The stream is opened inside the timed callable, unlike every other
        # adapter's setup. It has to be: a closed stream cannot be reopened to
        # write the same frames again, so hoisting it would leave the second
        # repeat measuring an error.
        stream = az.ZarrStream(settings)
        stream.append(data)
        stream.close()

    return Measured(write, _NOTE, extra={"target": str(path)})
