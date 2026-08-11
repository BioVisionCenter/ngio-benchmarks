"""Shared setup for the pyramid writers, and the audit of what they produced."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

from ngio_benchmarks.core.data import load_source

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec


def source(spec: ImageSpec, root: Path):
    """The array to build a pyramid from, in memory.

    Materialised out of the memory-mapped file before the timing starts. Every
    writer is handed the same in-memory array, so what is measured is the
    pyramid it builds and not how it got hold of its input.
    """
    import numpy as np

    return np.ascontiguousarray(load_source(root, spec))


def target(root: Path, impl: str, spec: ImageSpec) -> Path:
    """A fresh store for one writer, cleared if it already exists."""
    path = root / f"create_{impl}_{spec.name}.zarr"
    if path.exists():
        shutil.rmtree(path)
    return path


def fresh(path: Path) -> Callable[[], None]:
    """A `Measured.setup` that empties `path` before every timed run.

    Without it, a writer measured `repeats` times does not do the same work
    `repeats` times. The first run writes into an empty directory; every later
    one first has to dispose of the store the previous run left, and the writers
    disagree about how -- `zarr.open_group(mode="w")`, `overwrite=True`, an
    `rmtree` of a few thousand chunk files. That cost lands inside the timed
    region for runs 2..n and not for run 1, so it both inflates the number and
    makes it depend on `repeats`.

    `compare.io._ops.target` already refuses to let two implementations share a
    store, for the same reason at a different scale: writing over existing
    chunks is not the operation being measured.
    """

    def clear() -> None:
        shutil.rmtree(path, ignore_errors=True)

    return clear


def scale_factors(spec: ImageSpec, *, explicit_ones: bool = False) -> list[dict]:
    """Cumulative per-axis downsample factors for the levels below level 0.

    `[{"y": 2, "x": 2}, {"y": 4, "x": 4}, ...]` -- cumulative, i.e. relative to
    level 0 rather than to the previous level, which is what `ome-zarr-py` and
    `ngff-zarr` both mean by the argument.

    Per-axis dicts rather than bare ints, which is the point. A bare `2` means
    "halve every spatial axis" to both libraries, so passing ints silently
    produced a pyramid with half the voxels per level of the one ngio, bioio
    and ome-zarr-py build. Naming the axes removes the ambiguity.

    `explicit_ones` spells out the axes that are *not* halved, as `1`.
    `ome-zarr-py` reads a dict as "these axes, the rest untouched";
    `ngff-zarr` indexes every spatial dim and raises `KeyError` on a missing
    one. Same requested pyramid, two spellings, so the caller says which.

    The factor is fixed at 2 per level. Every writer here can express halving
    and several can express nothing else -- iohub's pyramid is a cascade of
    halvings by construction -- so a configurable factor would be a setting
    only some columns could honour, which is the situation this function exists
    to end.
    """
    halved = set(spec.downsample_axes)
    named = [a for a in spec.axes if a in halved or (explicit_ones and a in "zyx")]
    return [
        {axis: 2**step if axis in halved else 1 for axis in named}
        for step in range(1, spec.levels)
    ]


def _shapes(path: Path) -> list[tuple[int, ...]]:
    """The shape of every zarr array anywhere under `path`, largest first.

    Walked rather than listed one level deep, because the writers do not agree
    on where the levels go -- some put them at the store root (`0`, `1`), some
    prefix them (`s0`, `scale0/image`), some nest them under the array key they
    were given (`0/0`, `0/1`).
    """
    shapes = []
    for entry in path.rglob("*"):
        try:
            if entry.name == ".zarray":
                shapes.append(tuple(json.loads(entry.read_text())["shape"]))
            elif entry.name == "zarr.json":
                data = json.loads(entry.read_text())
                if data.get("node_type") == "array":
                    shapes.append(tuple(data["shape"]))
        except (OSError, ValueError, KeyError):
            continue
    # Sorted by size so level 0 comes first regardless of naming scheme; a
    # writer's own ordering is not something to trust when the point is to
    # check what it produced.
    shapes.sort(key=lambda s: -_volume(s))
    return shapes


def _volume(shape: tuple[int, ...]) -> int:
    """Number of elements in `shape`."""
    total = 1
    for extent in shape:
        total *= extent
    return total


def _render(shapes: list[tuple[int, ...]]) -> str:
    """Level shapes as one compact cell, e.g. `512x512 | 256x256 | 128x128`.

    Only the trailing three axes, and leading singletons dropped -- the whole
    5-tuple repeated per level does not fit a table cell, and the axes that
    differ between writers are always the spatial ones.
    """
    return " | ".join("x".join(str(e) for e in shape[-3:] if True) for shape in shapes)


def _level_zero(path: Path) -> dict:
    """The metadata document of the largest array under `path`.

    Level 0, found by size rather than by name, because the writers do not agree
    where it lives. Returns `{}` when nothing readable is there.
    """
    best, largest = {}, -1
    for entry in path.rglob("*"):
        if entry.name not in (".zarray", "zarr.json"):
            continue
        try:
            data = json.loads(entry.read_text())
        except (OSError, ValueError):
            continue
        if entry.name == "zarr.json" and data.get("node_type") != "array":
            continue
        shape = data.get("shape")
        if shape and _volume(tuple(shape)) > largest:
            best, largest = data, _volume(tuple(shape))
    return best


def _codec(meta: dict) -> str:
    """The compression the store actually used, named as its metadata names it.

    Read off disk for the same reason the pyramid is: what a writer was asked
    for and what it did are different claims, and only one of them is checkable.
    This column exists because five of the six writers used to ignore the
    requested codec entirely and fall back to their own default -- which is
    invisible in a timing and shows up only as a store that is a third smaller.

    zarr v2 puts one `compressor` object in `.zarray`; v3 puts a `codecs` list in
    `zarr.json`, where the compressor sits after `bytes` and, when sharded,
    inside the sharding codec's configuration.
    """
    if "compressor" in meta:  # zarr v2
        compressor = meta["compressor"]
        if not compressor:
            return "none"
        name = compressor.get("id", "?")
        cname = compressor.get("cname")
        level = compressor.get("clevel", compressor.get("level"))
        return f"{name}{f'/{cname}' if cname else ''}{f'/{level}' if level else ''}"

    codecs = meta.get("codecs") or []
    for codec in codecs:
        if codec.get("name") == "sharding_indexed":
            codecs = codec.get("configuration", {}).get("codecs", [])
            break
    for codec in codecs:
        name = codec.get("name")
        if name in ("bytes", "transpose", "crc32c", "sharding_indexed"):
            continue
        config = codec.get("configuration") or {}
        cname = config.get("cname")
        level = config.get("clevel", config.get("level"))
        return f"{name}{f'/{cname}' if cname else ''}{f'/{level}' if level else ''}"
    return "none" if codecs else ""


def _grid(meta: dict) -> tuple[str, str]:
    """Level 0's chunk shape and shard shape, as written.

    Two writers given the same chunk shape can still produce different stores if
    one of them silently dropped the sharding it was asked for -- which is a
    thing that happens, and is worth a column rather than a surprise.
    """
    if "chunks" in meta:  # zarr v2, which cannot shard
        return "x".join(str(c) for c in meta["chunks"]), ""

    grid = (meta.get("chunk_grid") or {}).get("configuration") or {}
    outer = grid.get("chunk_shape") or []
    inner = outer
    for codec in meta.get("codecs") or []:
        if codec.get("name") == "sharding_indexed":
            inner = (codec.get("configuration") or {}).get("chunk_shape") or outer
            break
    chunks = "x".join(str(c) for c in inner)
    shards = "x".join(str(c) for c in outer) if inner != outer else ""
    return chunks, shards


def _declared(path: Path) -> int:
    """How many datasets the store's own NGFF metadata declares."""
    for meta in (*path.rglob(".zattrs"), *path.rglob("zarr.json")):
        try:
            data = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        block = data.get("attributes", data).get("ome", data).get("multiscales")
        if block:
            return len(block[0].get("datasets", []))
    return 0


def audit(path: Path, spec: ImageSpec | None = None) -> dict[str, str]:
    """Report what a writer actually produced, read back off disk.

    Not taken from what the writer was asked for: the point of these columns is
    to catch a writer that quietly produced something else, and asking it what
    it did would not catch that. It is how `acquire-zarr` was caught writing one
    level too many, and how the xy-versus-xyz downsampling disagreement became
    visible instead of showing up only as an unexplained gap in the timings.

    Deliberately done by reading the store's own metadata with `json`, not by
    opening it with a library: the auditor must not be one of the contestants,
    and it has to behave identically inside all six environments.

    Three columns, because there are three distinct ways to differ:

    * `levels` -- how many arrays exist. When the arrays on disk and the NGFF
      metadata disagree, both numbers are reported: a store with five arrays
      declaring three is one readers will only ever see three of, which is a
      different fault from writing three.
    * `level_shapes` -- their shapes, so a pyramid that halved the wrong axes
      is visible even when the level count is right.
    * `pyramid` -- `as asked`, or the divergence spelled out.
    * `codec`, `chunks`, `shards` -- how level 0 was actually stored. Two
      writers can build the identical pyramid and still produce stores that
      differ by a third in size, and when they do it is one of these three.
    """
    if not path.exists():
        return {"levels": "0"}
    shapes = _shapes(path)
    written, declared = len(shapes), _declared(path)
    levels = str(written)
    if declared and declared != written:
        levels = f"{written} (metadata declares {declared})"

    meta = _level_zero(path)
    chunks, shards = _grid(meta)
    audited = {
        "levels": levels,
        "level_shapes": _render(shapes),
        "codec": _codec(meta),
        "chunks": chunks,
        "shards": shards,
    }
    if spec is not None:
        # Compared on the trailing three axes only. A writer whose data model
        # is 5D pads a 4D request with a leading singleton -- iohub does -- and
        # that is a reshape this suite asked for, not a pyramid difference.
        want = [s[-3:] for s in spec.level_shapes()]
        audited["pyramid"] = (
            "as asked"
            if [s[-3:] for s in shapes] == want
            else f"differs; asked for {_render(spec.level_shapes())}"
        )
    return audited
