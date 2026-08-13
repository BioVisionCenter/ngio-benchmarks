"""Shared instrument for the `reports/` harnesses: count what a write really does.

Every question these harnesses ask reduces to the same three measurements —
*how many reads did the store issue*, *how many bytes did they move*, and *did
the array end up holding the right pixels*. This module owns those, so the two
harnesses cannot drift into measuring subtly different things and then disagree
in print.

Not a harness itself and not under `ngio_benchmarks/`: everything in that
package has to import inside every peer library's environment, and this reaches
into zarr's store internals. The leading underscore marks it as the helper the
harnesses beside it share.

The one invariant every caller inherits: **snapshot the tally before reading the
array back.** Verification reads are reads, and folding them in turns a clean
zero into a confident false positive. `Tally.enabled` exists for exactly that —
flip it off once the write is done.
"""

from __future__ import annotations

import shutil
import threading
from collections import Counter
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.storage import LocalStore

from ngio_benchmarks.core.data import synthetic
from ngio_benchmarks.core.images import zarr_compressors

MB = 1024 * 1024

#: The pipeline zarr ships enabled, and the opt-in one. They read the store
#: through *different* methods -- `get` and `get_sync` -- so an instrument
#: counting one can report a confident zero while the other does the reading.
PIPELINES = {
    "batched": "zarr.core.codec_pipeline.BatchedCodecPipeline",
    "fused": "zarr.core.codec_pipeline.FusedCodecPipeline",
}


# --------------------------------------------------------------------------
# The instrument
# --------------------------------------------------------------------------


@dataclass
class Tally:
    """Store reads, counted. Shared by every instance a store spawns."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    enabled: bool = True
    calls: int = 0
    nbytes: int = 0
    per_key: Counter[str] = field(default_factory=Counter)

    def record(self, key: str, buf: Any) -> None:
        """Note one read, if counting is on."""
        size = 0 if buf is None else len(buf)
        with self.lock:
            if not self.enabled:
                return
            self.calls += 1
            self.nbytes += size
            self.per_key[key] += 1

    def reset(self) -> None:
        """Forget everything counted so far."""
        with self.lock:
            self.calls = self.nbytes = 0
            self.per_key.clear()

    def snapshot(self) -> dict[str, Any]:
        """Take this *before* reading the array back -- see the module docstring."""
        with self.lock:
            return {
                "calls": self.calls,
                "mib": self.nbytes / MB,
                "keys": len(self.per_key),
            }


class CountingStore(LocalStore):
    """`LocalStore` that tallies every read, on both pipelines' paths.

    `zarr.create_array(store=<instance>)` keeps this object's identity: a `Store`
    is returned as-is rather than rewrapped, and `create_array` opens with mode
    `"a"`, so the read-only branch is never taken. `with_read_only` is overridden
    anyway -- it constructs a *fresh* instance, which without this would carry a
    fresh counter and report a confident zero.
    """

    def __init__(
        self, root: Any, *, read_only: bool = False, tally: Tally | None = None
    ) -> None:
        """Wrap `root`, sharing `tally` with every store spawned from this one."""
        super().__init__(root, read_only=read_only)
        self.tally = tally if tally is not None else Tally()

    def with_read_only(self, read_only: bool = False) -> CountingStore:
        """A read-only twin that keeps counting into the same tally."""
        return type(self)(root=self.root, read_only=read_only, tally=self.tally)

    async def get(self, key, prototype=None, byte_range=None):  # type: ignore[override]
        """The async read path: `BatchedCodecPipeline`, the shipped default."""
        buf = await super().get(key, prototype, byte_range)
        self.tally.record(key, buf)
        return buf

    def get_sync(self, key, *, prototype=None, byte_range=None):  # type: ignore[override]
        """The sync read path: `FusedCodecPipeline`, and easy to forget."""
        buf = super().get_sync(key, prototype=prototype, byte_range=byte_range)
        self.tally.record(key, buf)
        return buf

    async def get_partial_values(self, prototype, key_ranges):  # type: ignore[override]
        """Byte-range reads, counted per key like any other."""
        pairs = list(key_ranges)
        bufs = await super().get_partial_values(prototype, pairs)
        for (key, _), buf in zip(pairs, bufs, strict=True):
            self.tally.record(key, buf)
        return bufs


# --------------------------------------------------------------------------
# Geometry -- asserted, never assumed
# --------------------------------------------------------------------------


def write_unit(array: zarr.Array) -> tuple[int, ...]:
    """The granularity zarr actually writes in: the shard, else the chunk.

    The single most misstated fact in this repo's history, so it gets a name.
    A write covering one whole unit skips the read; anything less read-modify-
    writes the unit. `array.shards` is `None` on an unsharded array, and on a
    sharded one it is a multiple of `array.chunks` -- so this expression is the
    whole rule.
    """
    return array.shards or array.chunks


def geometry(spec, block: tuple[int, ...] | None = None) -> dict[str, Any]:
    """What the write actually looks like against the target's write unit.

    `blocks_per_unit == 1` is the whole predicate for "a full write does no
    read-modify-write". Computed rather than stated, so the report's claim is
    scoped by arithmetic that travels with the specs instead of by prose that
    goes stale the moment someone adds an image.
    """
    block = block or spec.chunks
    unit = spec.shards or spec.chunks
    return {
        "block": block,
        "unit": unit,
        "sharded": spec.shards is not None,
        "blocks": prod(-(-s // b) for s, b in zip(spec.shape, block, strict=True)),
        "units": prod(-(-s // u) for s, u in zip(spec.shape, unit, strict=True)),
        "blocks_per_unit": prod(
            max(1, u // b) for u, b in zip(unit, block, strict=True)
        ),
        "exact": all(s % b == 0 for s, b in zip(spec.shape, block, strict=True)),
    }


def marked(spec) -> np.ndarray:
    """`synthetic`, made unique per z-plane, and never zero.

    Two properties the race arms cannot do without. `synthetic` broadcasts one
    plane across the leading axes, so without an offset every z-slice is
    identical and a lost update whose stale snapshot already held that content
    is invisible -- corruption would be undercounted, in the direction that
    flatters the hypothesis. And keeping every value non-zero makes "still holds
    the fill value" unambiguously "never written".
    """
    base = synthetic(spec.shape, spec.dtype, spec.seed).astype("uint32")
    z = spec.axes.index("z")
    offsets = np.arange(spec.shape[z], dtype="uint32") * 257
    shaped = offsets.reshape([-1 if i == z else 1 for i in range(base.ndim)])
    return np.ascontiguousarray(((base + shaped) % 65535 + 1).astype(spec.dtype))


def region(spec, op: str) -> tuple[slice, ...]:
    """The same index `compare.io._ops` gives this op, restated locally.

    Restated rather than imported: these scripts have to keep working as a
    record of what was measured even if the suite's definitions move.
    """
    if op.endswith("_full"):
        return tuple(slice(None) for _ in spec.shape)
    start = 0 if op.endswith("_aligned") else spec.chunks[-1] // 2 + 1
    stop = start + spec.roi_size
    leading = tuple(slice(None) for _ in spec.shape[:-2])
    return (*leading, slice(start, stop), slice(start, stop))


def create(
    path: Path,
    spec,
    tally: Tally,
    *,
    shape: tuple[int, ...] | None = None,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
) -> Any:
    """A fresh target, wired to the counting store.

    The three overrides exist for the transform shapes, where the target is not
    the source: a z-reduction and a half-scale pyramid level both need a
    differently shaped array built from the same spec's dtype and codecs.

    An override is honoured only when the spec already asks for that property,
    so a caller passing a shard shape cannot silently turn an unsharded arm into
    a sharded one -- and, the way round that actually bit, omitting the override
    cannot silently turn the *sharded* arm unsharded. `spec.shards is None` is
    what makes an arm unsharded; nothing here may override it in either
    direction.
    """
    if path.exists():
        shutil.rmtree(path)
    return zarr.create_array(
        store=CountingStore(path, tally=tally),
        shape=shape or spec.shape,
        dtype=spec.dtype,
        chunks=chunks or spec.chunks,
        shards=(shards or spec.shards) if spec.shards is not None else None,
        compressors=zarr_compressors(spec),
        zarr_format=spec.zarr_format,
        overwrite=True,
    )


def store_bytes(array: zarr.Array) -> int:
    """Bytes the array occupies on disk.

    Worth recording next to every write: a store markedly *smaller* than its
    peers for the same op is this repo's earliest signal of a corrupting arm,
    because the blocks that never survived also never got compressed and
    written. It reads as "compressed better" until you check the checksum.
    """
    root = Path(array.store.root)  # type: ignore[attr-defined]
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """A markdown table, ready to paste into the report."""
    head = "| " + " | ".join(title for title, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |"
        for row in rows
    ]
    return "\n".join([head, rule, *body])


def fmt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round the numbers a reader actually compares.

    Every key is optional, and every value may be a string instead of a number:
    the two harnesses record overlapping but different columns, and an arm whose
    reads cannot be counted -- because it writes through Rust, or through a
    store this module never wrapped -- reports `"n/a"` there. Passing that
    through untouched is the point; formatting it as `0.0` would read as
    "issued none" rather than "not measurable here".
    """

    def num(row: dict[str, Any], key: str) -> bool:
        return isinstance(row.get(key), (int, float)) and not isinstance(
            row.get(key), bool
        )

    out = []
    for row in rows:
        shown = dict(row)
        if num(row, "wall"):
            shown["wall"] = f"{row['wall'] * 1000:.0f} ms"
        if num(row, "cpu_wall"):
            shown["cpu_wall"] = f"{row['cpu_wall']:.2f}x"
        if num(row, "mib"):
            shown["mib"] = f"{row['mib']:.1f}"
        if num(row, "rss"):
            shown["rss"] = f"{row['rss']:.0f} MB"
        if num(row, "bytes"):
            shown["bytes"] = f"{row['bytes'] / MB:.1f} MiB"
        if num(row, "wrong"):
            shown["wrong"] = f"{row['wrong']:,}"
            if num(row, "total") and row["total"]:
                shown["pct"] = f"{100 * row['wrong'] / row['total']:.1f}%"
        out.append(shown)
    return out
