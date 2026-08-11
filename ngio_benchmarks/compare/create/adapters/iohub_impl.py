"""iohub: an HCS-shaped writer, measured on a single position.

Written for high-content screening, so its natural unit is a plate of positions
rather than one image. Measured here on a single FOV, which is the only shape
comparable with the rest of the table -- but worth knowing when reading the
number, because some of what it costs is machinery for a case this suite is not
exercising.

The detail that mattered most here is not iohub's own code. Its registry default
is `zarrs-python`, the Rust codec pipeline, while every other column in this
table encodes through zarr-python's. That is a real and interesting difference,
but it was inside one column labelled with the library's name, so it read as
iohub being fast. It is now `[options.iohub] implementation`, defaulting to the
library's own choice and recorded on every row.

Requires Python 3.12, while ngio supports 3.11. Its environment therefore pins
its own interpreter rather than inheriting the parent's, which is the whole
argument for one environment per implementation stated as a single line of
configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare import _pipeline
from ngio_benchmarks.compare.create import _ops
from ngio_benchmarks.core.measure import Measured, Unsupported

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "iohub"
DISTRIBUTION = "iohub"
#: `tensorstore` is not iohub's dependency; it is what its third registered
#: implementation imports. Installed here so the `implementation` option can
#: actually reach all three, on the same reasoning as ngff-zarr's extras.
REQUIRES = ("iohub>=0.3.10", "tensorstore>=0.1.85", "numpy>=2")
SUPPORTS = frozenset({"create_pyramid"})
#: `open_ome_zarr` writes NGFF 0.4 or 0.5.
FORMATS = frozenset({2, 3})
PYTHON = "3.12"
REPEATS = 3

#: Canonical name -> `compute_pyramid(method=...)`.
#:
#: `stride` is `data[::2]`, which is what nearest means here. `median`, `min`,
#: `max` and `mode` have no canonical name and are reachable through
#: `[options.iohub] method_native`.
METHODS = {
    "default": "mean",
    "mean": "mean",
    "nearest": "stride",
}

OPTIONS = {
    #: Which zarr stack does the encoding. Open, because iohub resolves the
    #: name through a plugin registry a config may have extended.
    "implementation": ["zarr-python", "zarrs-python", "tensorstore"],
}


def excludes_pipeline(pipeline: str, values: dict) -> str:
    """Why the suite's `pipeline` axis does not reach iohub, whatever it holds.

    The only zarr-python writer in this table that declines the axis, and not
    because the pipeline could not reach it -- under `implementation
    ="zarr-python"` it would. It declines because iohub already answers this
    question through its own registry, one option above, and asking it twice in
    two vocabularies produces a row (`implementation=zarr-python
    pipeline=zarrs`) that is zarrs by a second route and reads as neither.

    Its `zarrs-python` entry is iohub's own integration rather than the codec
    pipeline swap, which is a difference worth keeping visible.
    """
    if pipeline != _pipeline.DEFAULT:
        return (
            "iohub picks its zarr stack through its own registry; sweep "
            "[options.iohub] implementation instead"
        )
    return ""


def _tczyx(spec: ImageSpec, data):
    """Reshape to the (T, C, Z, Y, X) an iohub position always holds.

    iohub's FOV layout is 5D by definition, so a 4D spec is given a leading
    singleton time axis rather than being reported unsupported. The pixels and
    the pyramid are unchanged, so the column stays comparable -- but the note
    says so, because a reshape the other writers did not need is a difference
    this suite introduced.
    """
    missing = 5 - len(spec.shape)
    if missing <= 0:
        return data, ""
    return (
        data.reshape((1,) * missing + tuple(spec.shape)),
        f"input reshaped to 5D ({missing} leading axis added)",
    )


def build(
    op: str,
    spec: ImageSpec,
    root: Path,
    *,
    method: str = "default",
    method_native: str | None = None,
    implementation: str | None = None,
    **options: str,
) -> Measured:
    """Create a position, write level 0, compute the pyramid."""
    from iohub import open_ome_zarr

    filter_ = method_native or METHODS[method]

    data, note = _tczyx(spec, _ops.source(spec, root))
    path = _ops.target(root, NAME, spec)
    downsample = set(spec.downsample_axes)
    channels = (
        [f"Channel {i + 1}" for i in range(spec.shape[0])]
        if "c" in spec.axes
        else ["Channel 1"]
    )
    chunks = (1,) * (5 - len(spec.chunks)) + tuple(spec.chunks)
    shards_ratio = None
    if spec.shards:
        ratio = tuple(
            shard // chunk
            for shard, chunk in zip(spec.shards, spec.chunks, strict=True)
        )
        shards_ratio = (1,) * (5 - len(ratio)) + ratio

    opened: dict[str, object] = {}
    if implementation is not None:
        opened["implementation"] = implementation
    config = _config(spec, implementation)
    if config is not None:
        opened["implementation_config"] = config

    def create() -> None:
        with open_ome_zarr(
            str(path),
            layout="fov",
            mode="w",
            channel_names=channels,
            version="0.4" if spec.zarr_format == 2 else "0.5",
            **opened,
        ) as position:
            position.create_image("0", data, chunks=chunks, shards_ratio=shards_ratio)
            if spec.levels > 1:
                # `dims` pinned: iohub defaults to {"z", "y", "x"}, so without
                # this it halves z as well and builds a pyramid with half the
                # voxels per level of the one ngio, bioio and ome-zarr-py
                # build -- a difference that reads as iohub being fast.
                position.initialize_pyramid(levels=spec.levels, dims=downsample)
                position.compute_pyramid(method=filter_, dims=downsample)

    return Measured(
        create,
        note,
        extra={
            "target": str(path),
            "method_native": filter_,
            "implementation": implementation or _default_implementation(),
        },
        setup=_ops.fresh(path),
    )


def _default_implementation() -> str:
    """Whichever backend iohub picked for itself, recorded rather than assumed.

    Read from the registry rather than hard-coded: which stack is default is
    exactly the kind of thing that changes in a point release, and a column that
    said `zarrs-python` because this file did would be worse than no column.
    """
    try:
        from iohub.core.registry import _default

        return str(_default)
    except Exception:  # pragma: no cover - registry is private
        return "library default"


def _config(spec: ImageSpec, implementation: str | None):
    """The `implementation_config` carrying this spec's codec, or None.

    iohub takes compression through a pydantic config object rather than a codec
    instance, and it is blosc-only: `cname`, `clevel` and `shuffle` are the whole
    surface. `zstd` and `lz4` map onto it exactly; `none` does not -- there is no
    way to ask for an uncompressed array -- so that one is reported rather than
    silently written compressed.
    """
    if spec.compressors == "auto":
        return None
    if spec.compressors == "none":
        raise Unsupported(
            "iohub always compresses: its config exposes blosc cname, clevel "
            "and shuffle, with no way to write an uncompressed array"
        )
    from iohub.core.config import CompressorConfig, TensorStoreConfig, ZarrConfig

    cname = "zstd" if spec.compressors in ("zstd", "blosc") else spec.compressors
    compressor = CompressorConfig(cname=cname, clevel=1, shuffle="bitshuffle")
    if implementation == "tensorstore":
        return TensorStoreConfig(compressor=compressor)
    return ZarrConfig(compressor=compressor)
