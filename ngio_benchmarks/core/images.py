"""Named image specs: the vocabulary of the `image` axis.

Every block that touches real data sweeps a closed `image` axis whose labels
name entries in the registry below. A config file may add entries with an
`[images.<name>]` table.

That is deliberately *not* a hole in the rule that a config can only replace an
axis a block declares. The block still declares `image`, and it is still
closed; the table adds a value to the registry the axis draws from. The
distinction matters because a spec is a bundle of interdependent settings --
shards imply zarr v3, a shard shape must tile the chunk shape -- and a bundle
wants validating once, by name, not re-derived at every use site.

The names are the unit of comparison, so keep them meaningful. Two entries that
differ only in one field are worth having when that field is the experiment
(`chunks256` against `chunks64`); two that differ in five are worth having only
if someone can say what the pair is for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

#: Axis names by rank, following the OME-NGFF ordering. A spec may override
#: them, but the default is right for every shape anyone sweeps here.
_AXES_BY_RANK = {
    2: ("y", "x"),
    3: ("z", "y", "x"),
    4: ("c", "z", "y", "x"),
    5: ("t", "c", "z", "y", "x"),
}

#: What each `compressors` label means. `auto` defers to whatever the writing
#: library considers sensible, which is itself a thing worth comparing -- the
#: peers do not agree on it.
CODECS = ("auto", "none", "zstd", "blosc", "lz4")


class ImageSpec(NamedTuple):
    """One named image, fully resolved.

    Both halves of the project read this: `internal` hands it to ngio, and each
    comparison adapter translates it into its own library's vocabulary. Keeping
    it a plain record of primitives -- rather than, say, a dict of
    `create_empty_ome_zarr` kwargs -- is what makes that translation possible.
    """

    name: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None = None
    dtype: str = "uint16"
    compressors: str = "auto"
    zarr_format: int = 3
    levels: int = 3
    #: Which axes are halved at each pyramid level. Empty means the default,
    #: `("y", "x")`.
    #:
    #: This is not a detail. The writers in `compare-create` disagree about it
    #: out of the box -- ngio, ome-zarr-py and bioio downsample xy only, while
    #: ngff-zarr and iohub also halve z -- which means their default pyramids
    #: differ by 2x in voxels at every level and their timings are not
    #: comparable at all. Pinning it here is what makes the column a comparison
    #: of implementations rather than of pyramid shapes.
    downsample: tuple[str, ...] = ()
    axes_names: tuple[str, ...] = ()
    pixelsize: tuple[float, float] = (0.65, 0.65)
    #: Edge length of the ROI the `read_roi_*` / `write_roi_*` operations use,
    #: in pixels, on the two trailing axes.
    roi_size: int = 512
    seed: int = 0

    @property
    def axes(self) -> tuple[str, ...]:
        """Axis names, defaulted from the rank."""
        return self.axes_names or _AXES_BY_RANK[len(self.shape)]

    @property
    def downsample_axes(self) -> tuple[str, ...]:
        """Axes halved per level, defaulted to xy.

        xy-only is the default because it is what most bioimaging tools mean by
        a pyramid: z is usually already coarse relative to xy, and halving it
        throws away the axis with the least to spare. It is also the majority
        behaviour among the writers being compared -- but the point of the
        field is that the minority is now a setting rather than a surprise.
        """
        return self.downsample or ("y", "x")

    def level_shapes(self) -> list[tuple[int, ...]]:
        """The shape of every pyramid level, level 0 first.

        The single definition of what pyramid this spec asks for. Each adapter
        translates it into whatever its library accepts -- explicit shapes,
        cumulative per-axis factors, a set of axis names -- and the audit reads
        the real shapes back off disk to check they agree.
        """
        halved = set(self.downsample_axes)
        shapes = []
        for step in range(self.levels):
            factor = 2**step
            shapes.append(
                tuple(
                    max(extent // factor, 1) if name in halved else extent
                    for name, extent in zip(self.axes, self.shape, strict=True)
                )
            )
        return shapes

    @property
    def ngff_version(self) -> str:
        """The NGFF version implied by the zarr format.

        NGFF 0.4 maps to zarr format 2, which cannot shard; 0.5 maps to zarr 3.
        The two are not independent knobs, so the spec carries one of them.
        """
        return "0.4" if self.zarr_format == 2 else "0.5"

    @property
    def nbytes(self) -> int:
        """Uncompressed size of level 0."""
        size = int(_ITEMSIZE[self.dtype])
        for extent in self.shape:
            size *= extent
        return size

    def roi_offset(self, *, aligned: bool) -> int:
        """Start offset on the trailing axes for an aligned/straddling read.

        The straddling offset is derived rather than configured: it has exactly
        one job, which is to land inside a chunk instead of on its boundary.
        Half a chunk plus one does that for any chunk shape, whereas a constant
        borrowed from one layout would quietly become aligned under another.
        """
        return 0 if aligned else self.chunks[-1] // 2 + 1

    def to_dict(self) -> dict[str, Any]:
        """Render back to plain TOML types, for a child process's config."""
        data: dict[str, Any] = {
            "shape": list(self.shape),
            "chunks": list(self.chunks),
            "dtype": self.dtype,
            "compressors": self.compressors,
            "zarr_format": self.zarr_format,
            "levels": self.levels,
            "downsample": list(self.downsample_axes),
            "axes_names": list(self.axes),
            "pixelsize": list(self.pixelsize),
            "roi_size": self.roi_size,
            "seed": self.seed,
        }
        if self.shards is not None:
            data["shards"] = list(self.shards)
        return data


_ITEMSIZE = {"uint8": 1, "int8": 1, "uint16": 2, "int16": 2, "uint32": 4, "float32": 4}


#: The shipped vocabulary. Sizes are chosen so that `small` is a smoke test,
#: `medium` is the default working size, and `large` is where a mode that scales
#: with the data starts to hurt.
BUILTIN: dict[str, ImageSpec] = {
    "small": ImageSpec(
        name="small", shape=(1, 16, 512, 512), chunks=(1, 1, 256, 256), roi_size=256
    ),
    "medium": ImageSpec(
        name="medium", shape=(1, 64, 1024, 1024), chunks=(1, 1, 256, 256), levels=4
    ),
    "large": ImageSpec(
        name="large", shape=(1, 256, 1024, 1024), chunks=(1, 1, 256, 256), levels=5
    ),
    # Same bytes as `medium`, laid out four different ways. This quartet is the
    # point of having a registry at all: the layout is the experiment, and the
    # only honest way to compare layouts is to hold everything else equal.
    "chunks64": ImageSpec(
        name="chunks64", shape=(1, 64, 1024, 1024), chunks=(1, 1, 64, 64), levels=4
    ),
    "chunks512": ImageSpec(
        name="chunks512", shape=(1, 64, 1024, 1024), chunks=(1, 1, 512, 512), levels=4
    ),
    "sharded": ImageSpec(
        name="sharded",
        shape=(1, 64, 1024, 1024),
        chunks=(1, 1, 128, 128),
        shards=(1, 4, 1024, 1024),
        levels=4,
    ),
    "uncompressed": ImageSpec(
        name="uncompressed",
        shape=(1, 64, 1024, 1024),
        chunks=(1, 1, 256, 256),
        compressors="none",
        levels=4,
    ),
    # zarr format 2. Not a curiosity: `z5py` reads nothing else, so without a
    # v2 entry in the registry it could never appear in a comparison at all.
    "v2": ImageSpec(
        name="v2",
        shape=(1, 64, 1024, 1024),
        chunks=(1, 1, 256, 256),
        zarr_format=2,
        levels=4,
    ),
}


def _ints(where: str, value: Any, *, rank: int | None = None) -> tuple[int, ...]:
    """Validate a TOML list of positive ints."""
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{where} must be a non-empty list of integers")
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise SystemExit(f"{where} must hold positive integers, got {item!r}")
    if rank is not None and len(value) != rank:
        raise SystemExit(f"{where} must have {rank} entries to match `shape`")
    return tuple(value)


def from_toml(name: str, table: Any, where: Path) -> ImageSpec:
    """Build one `ImageSpec` from an `[images.<name>]` table.

    Validated eagerly and completely. A spec is used by every implementation in
    a comparison, so a bad one does not fail a case -- it fails a column, and
    only after the environments have been installed.
    """
    if not isinstance(table, dict):
        raise SystemExit(f"{where}: [images.{name}] must be a table")
    unknown = sorted(set(table) - set(ImageSpec._fields) - {"name"})
    if unknown:
        raise SystemExit(
            f"{where}: images.{name} has unknown key {unknown[0]!r}; "
            f"valid keys are {', '.join(f for f in ImageSpec._fields if f != 'name')}"
        )

    if "shape" not in table:
        raise SystemExit(f"{where}: images.{name} must set `shape`")
    shape = _ints(f"images.{name}.shape", table["shape"])
    if len(shape) not in _AXES_BY_RANK:
        raise SystemExit(
            f"{where}: images.{name}.shape must have 2 to 5 entries, got {len(shape)}"
        )

    if "chunks" not in table:
        raise SystemExit(f"{where}: images.{name} must set `chunks`")
    chunks = _ints(f"images.{name}.chunks", table["chunks"], rank=len(shape))

    shards = None
    if "shards" in table:
        shards = _ints(f"images.{name}.shards", table["shards"], rank=len(shape))

    zarr_format = table.get("zarr_format", 3)
    if zarr_format not in (2, 3):
        raise SystemExit(f"{where}: images.{name}.zarr_format must be 2 or 3")
    if shards is not None and zarr_format == 2:
        # Caught here rather than at write time, where it would surface as a
        # zarr error four layers down naming neither the spec nor the file.
        raise SystemExit(
            f"{where}: images.{name} sets `shards` with zarr_format 2, which "
            "cannot shard. Sharding needs zarr_format 3 (NGFF 0.5)."
        )
    if shards is not None:
        bad = [(s, c) for s, c in zip(shards, chunks, strict=True) if s % c or s < c]
        if bad:
            raise SystemExit(
                f"{where}: images.{name} shard shape {shards} is not a whole "
                f"number of chunks {chunks} on every axis"
            )

    dtype = table.get("dtype", "uint16")
    if dtype not in _ITEMSIZE:
        raise SystemExit(
            f"{where}: images.{name}.dtype must be one of {', '.join(_ITEMSIZE)}"
        )

    compressors = table.get("compressors", "auto")
    if compressors not in CODECS:
        raise SystemExit(
            f"{where}: images.{name}.compressors must be one of {', '.join(CODECS)}"
        )

    axes_names = tuple(table.get("axes_names", ())) or _AXES_BY_RANK[len(shape)]
    if len(axes_names) != len(shape):
        raise SystemExit(f"{where}: images.{name}.axes_names must match `shape`")

    downsample = tuple(table.get("downsample", ()))
    unknown_axes = [a for a in downsample if a not in axes_names]
    if unknown_axes:
        raise SystemExit(
            f"{where}: images.{name}.downsample names {unknown_axes[0]!r}, which "
            f"is not an axis of this image; axes are {', '.join(axes_names)}"
        )
    if downsample and not set(downsample) <= {"z", "y", "x"}:
        # Downsampling a channel or time axis is not a pyramid, and none of the
        # writers being compared will do it -- so it is rejected here rather
        # than producing five different interpretations of the request.
        raise SystemExit(
            f"{where}: images.{name}.downsample may only name spatial axes (z, y, x)"
        )

    pixelsize = tuple(float(v) for v in table.get("pixelsize", (0.65, 0.65)))
    if len(pixelsize) != 2:
        raise SystemExit(f"{where}: images.{name}.pixelsize must be [y, x]")

    roi_size = table.get("roi_size", min(512, *shape[-2:]))
    if not isinstance(roi_size, int) or roi_size < 1 or roi_size > min(shape[-2:]):
        raise SystemExit(
            f"{where}: images.{name}.roi_size must be between 1 and "
            f"{min(shape[-2:])} (the smaller trailing extent)"
        )

    return ImageSpec(
        name=name,
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        compressors=compressors,
        zarr_format=zarr_format,
        levels=int(table.get("levels", 3)),
        downsample=downsample,
        axes_names=axes_names,
        pixelsize=pixelsize,  # type: ignore[arg-type]
        roi_size=roi_size,
        seed=int(table.get("seed", 0)),
    )


def registry(extra: Mapping[str, ImageSpec] | None = None) -> dict[str, ImageSpec]:
    """The built-in specs, plus any a config file added.

    A config entry with a built-in name replaces it. That is the one way to
    resize `medium` without inventing a name nobody else will recognise, and it
    is safe because the CSV records the spec's fields, not just its label.
    """
    return {**BUILTIN, **(extra or {})}


def zarr_compressors(spec: ImageSpec) -> Any:
    """Translate `spec.compressors` into what `zarr.create_array` wants.

    Imported lazily: this module is loaded in every child environment, and an
    adapter such as `acquire-zarr` has no reason to hold zarr.
    """
    if spec.compressors == "auto":
        return "auto"
    if spec.compressors == "none":
        return None
    if spec.zarr_format == 2:
        import numcodecs

        if spec.compressors == "zstd":
            return numcodecs.Zstd()
        return numcodecs.Blosc(cname="lz4" if spec.compressors == "lz4" else "zstd")

    from zarr.codecs import BloscCodec, ZstdCodec

    if spec.compressors == "zstd":
        return ZstdCodec()
    return BloscCodec(cname="lz4" if spec.compressors == "lz4" else "zstd")
