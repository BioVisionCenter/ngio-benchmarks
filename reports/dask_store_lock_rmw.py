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

The instrument itself lives in `_probe.py`, shared with
`lazy_write_strategies.py`, so the two harnesses cannot drift into measuring
subtly different things and then disagree in print.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import dask
import dask.array as da
import numpy as np
import zarr
from dask.utils import SerializableLock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ngio_benchmarks.core.images import BUILTIN
from reports._probe import (
    PIPELINES,
    Tally,
    create,
    fmt,
    geometry,
    marked,
    region,
    table,
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
