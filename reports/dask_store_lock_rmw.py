"""Does `da.store` into a zarr array read-modify-write, and can it lose updates?

The harness behind `reports/dask-sharded-write-races.md`. It exists because the
answer is "it depends on the geometry", and every prose statement of *which*
geometry -- including three in this repo, and one upstream in ngio -- got the
predicate subtly wrong. A predicate this easy to misstate should be measured.

    uv run python reports/dask_store_lock_rmw.py --quick
    uv run python reports/dask_store_lock_rmw.py

Three tables come out, ready to paste into the report: store reads per arm, the
lost-update sweep, and the cost attribution.

Deliberately not under `ngio_benchmarks/`: everything in that package has to
import inside every peer environment, and this needs dask and zarr's internals
at once. Deliberately not a test either -- it asserts a property of zarr-python,
not of this repo, so as a test it would turn red the day zarr changes its
complete-chunk fast path, which is news rather than breakage.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import dask
import dask.array as da
import numpy as np
import zarr
from dask.utils import SerializableLock
from zarr.storage import LocalStore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ngio_benchmarks.core.data import synthetic
from ngio_benchmarks.core.images import BUILTIN, zarr_compressors

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
        """Take this *before* reading the array back -- see `run_arm`."""
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

    Two properties the race arm cannot do without. `synthetic` broadcasts one
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

    Restated rather than imported: this script has to keep working as a record
    of what was measured even if the suite's definitions move.
    """
    if op.endswith("_full"):
        return tuple(slice(None) for _ in spec.shape)
    start = 0 if op.endswith("_aligned") else spec.chunks[-1] // 2 + 1
    stop = start + spec.roi_size
    leading = tuple(slice(None) for _ in spec.shape[:-2])
    return (*leading, slice(start, stop), slice(start, stop))


def create(path: Path, spec, tally: Tally) -> Any:
    """A fresh target, wired to the counting store."""
    if path.exists():
        shutil.rmtree(path)
    return zarr.create_array(
        store=CountingStore(path, tally=tally),
        shape=spec.shape,
        dtype=spec.dtype,
        chunks=spec.chunks,
        shards=spec.shards,
        compressors=zarr_compressors(spec),
        zarr_format=spec.zarr_format,
        overwrite=True,
    )


# --------------------------------------------------------------------------
# Arm 1 -- how many reads does a write issue?
# --------------------------------------------------------------------------


def run_arm(
    root: Path,
    spec,
    op: str,
    *,
    label: str,
    block: tuple[int, ...] | None = None,
    lock: Any = False,
    numpy_setitem: bool = False,
    workers: int | None = None,
    trials: int = 1,
) -> list[dict[str, Any]]:
    """One arm: write, snapshot the counter, then verify. In that order.

    The order is the whole correctness of the instrument. Reading the array back
    issues reads of its own, and folding them in turned a clean zero into a
    confident false positive the first time this was validated.
    """
    source = marked(spec)
    index = region(spec, op)
    expected = np.ascontiguousarray(source[index])
    geom = geometry(spec, block)
    out = []

    for trial in range(trials):
        tally = Tally()
        target = root / f"{label.replace(' ', '_')}_{trial}.zarr"
        array = create(target, spec, tally)
        tally.reset()

        start, start_cpu = time.perf_counter(), time.process_time()
        if numpy_setitem:
            array[index] = expected
        else:
            patch = da.from_array(expected, chunks=block or spec.chunks)
            scheduler = {"scheduler": "threads"}
            if workers:
                scheduler["num_workers"] = workers
            with dask.config.set(**scheduler):
                da.store(patch, array, regions=index, lock=lock)
        wall = time.perf_counter() - start
        cpu = time.process_time() - start_cpu

        # BEFORE the readback. Verification reads are reads.
        counted = tally.snapshot()
        tally.enabled = False

        back = array[index]
        wrong = int((back != expected).sum())
        out.append(
            {
                "label": label,
                "trial": trial,
                "wall": wall,
                "cpu_wall": cpu / wall if wall else float("nan"),
                **counted,
                "wrong": wrong,
                "total": int(expected.size),
                **geom,
            }
        )
        shutil.rmtree(target, ignore_errors=True)
    return out


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
    """Round the numbers a reader actually compares."""
    return [
        {
            **row,
            "wall": f"{row['wall'] * 1000:.0f} ms",
            "cpu_wall": f"{row['cpu_wall']:.2f}x",
            "mib": f"{row['mib']:.1f}",
            "wrong": f"{row['wrong']:,}",
            "pct": f"{100 * row['wrong'] / row['total']:.1f}%",
        }
        for row in rows
    ]


READ_COLUMNS = [
    ("arm", "label"),
    ("dask blocks", "blocks"),
    ("write units", "units"),
    ("blocks/unit", "blocks_per_unit"),
    ("store reads", "calls"),
    ("keys read", "keys"),
    ("MiB read", "mib"),
    ("wrong elements", "wrong"),
]


def main(argv: list[str] | None = None) -> int:
    """Run every arm and print the three tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="quarter the z depth; the sharded arms move GiB at full size",
    )
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args(argv)

    def shrink(spec):
        if not args.quick:
            return spec
        z = spec.axes.index("z")
        shape = list(spec.shape)
        shape[z] = max(spec.shards[z] if spec.shards else 1, shape[z] // 4)
        return spec._replace(shape=tuple(shape))

    plain, shard = shrink(BUILTIN["medium"]), shrink(BUILTIN["sharded"])
    root = Path(mkdtemp(prefix="rmw-"))
    print(f"# working in {root}\n")

    try:
        print("## 1. Store reads issued by a write\n")
        arms: list[dict[str, Any]] = []
        for pipeline, path in PIPELINES.items():
            with zarr.config.set({"codec_pipeline.path": path}):
                arms += run_arm(
                    root, plain, "write_full", label=f"unsharded full [{pipeline}]"
                )
                arms += run_arm(
                    root,
                    plain,
                    "write_roi_straddling",
                    label=f"unsharded straddling [{pipeline}]",
                )
                arms += run_arm(
                    root,
                    plain,
                    "write_roi_aligned",
                    label=f"unsharded aligned [{pipeline}]",
                )
                arms += run_arm(
                    root, shard, "write_full", label=f"sharded full [{pipeline}]"
                )
                arms += run_arm(
                    root,
                    shard,
                    "write_full",
                    block=shard.shards,
                    label=f"sharded full, blocks=shard [{pipeline}]",
                )
                arms += run_arm(
                    root,
                    shard,
                    "write_full",
                    numpy_setitem=True,
                    label=f"sharded full, numpy setitem [{pipeline}]",
                )
        print(table(fmt(arms), READ_COLUMNS))

        print("\n## 2. Lost updates, by worker count\n")
        races: list[dict[str, Any]] = []
        for name, spec in (("unsharded", plain), ("sharded", shard)):
            for lock_name, lock in (("False", False), ("SerializableLock", None)):
                for workers in (1, 2, 4, 8):
                    races += run_arm(
                        root,
                        spec,
                        "write_full",
                        label=f"{name} lock={lock_name} workers={workers}",
                        lock=SerializableLock() if lock is None else lock,
                        workers=workers,
                        trials=args.trials,
                    )
        print(
            table(
                fmt(races),
                [
                    ("arm", "label"),
                    ("trial", "trial"),
                    ("store reads", "calls"),
                    ("MiB read", "mib"),
                    ("wrong elements", "wrong"),
                    ("of total", "pct"),
                ],
            )
        )

        print("\n## 3. What the lock costs, and what it buys\n")
        print(
            table(
                fmt(races),
                [
                    ("arm", "label"),
                    ("wall", "wall"),
                    ("cpu/wall", "cpu_wall"),
                    ("store reads", "calls"),
                    ("MiB read", "mib"),
                    ("wrong elements", "wrong"),
                ],
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
