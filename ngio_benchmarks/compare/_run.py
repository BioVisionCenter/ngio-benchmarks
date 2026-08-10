"""The child side of a comparison: one implementation, all of its cases.

Runs inside the isolated environment holding exactly that implementation. The
parent has already decided which cases this adapter can express, built any
shared fixture, and written the whole instruction set into the job -- so there
is nothing to negotiate here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from ngio_benchmarks.core import images as images_mod
from ngio_benchmarks.core.axes import Case
from ngio_benchmarks.core.config import DEFAULT_REPEATS
from ngio_benchmarks.core.measure import (
    FAILED,
    UNSUPPORTED,
    Measured,
    Result,
    Unsupported,
    measure,
    process_peak_mb,
)
from ngio_benchmarks.core.output import interpreter, version

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from ngio_benchmarks.core.job import Job


def environment(adapter: ModuleType, label: str) -> dict[str, str]:
    """Which versions produced these numbers.

    Recorded per row rather than once per file. A comparison table without the
    resolved versions beside it is not a result anyone can act on -- and since
    each implementation resolves its own dependency tree, `zarr` can differ
    between two columns of the same run.
    """
    package = getattr(adapter, "DISTRIBUTION", None) or getattr(adapter, "NAME", label)
    return {
        "impl": label,
        "impl_version": version(package),
        "zarr": version("zarr"),
        # Read at the very end of the child, so it is this implementation's
        # peak across all of its cases -- coarse, but it counts the native
        # allocations `tracemalloc` cannot see.
        "proc_peak_mb": f"{process_peak_mb():.1f}",
        **interpreter(),
    }


def _checksum(value: object) -> str:
    """A short digest of what a read returned.

    Recorded so that "every implementation read the same bytes" is a column
    anyone can check rather than an assumption the suite makes. Computed once,
    outside the timing.
    """
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.blake2s(array.tobytes(), digest_size=6).hexdigest()


def _stored_bytes(path: Path) -> str:
    """Total size on disk of a store a write produced.

    A writer that is fast because it wrote less is not a faster writer, and
    without this column the table cannot tell the two apart.
    """
    if not path.exists():
        return ""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return str(total)


def run(
    job: Job,
    impls: dict[str, str],
    audit: Callable[..., dict[str, str]] | None = None,
) -> list[Result]:
    """Measure every case this job names, in the current environment.

    `audit` inspects whatever a case wrote and returns extra columns for it --
    how many pyramid levels really exist, say. Run once, after the timing, and
    against the store on disk rather than against what the adapter says it did,
    because catching a writer that quietly produced less is the entire reason
    the column exists.
    """
    import importlib

    adapter = importlib.import_module(impls[job.impl])
    registry = images_mod.registry(job.images)
    root = Path(job.root)
    repeats = job.repeats or getattr(adapter, "REPEATS", DEFAULT_REPEATS)

    results: list[Result] = []
    for entry in job.ops:
        op, image = entry.split("\t")
        spec = registry[image]
        case = Case(op, {"image": image}, {})
        extra = {"image": image, "zarr_format": str(spec.zarr_format)}
        try:
            measured: Measured = adapter.build(op, spec, root)
        except Unsupported as error:
            results.append(Result.blank(case, UNSUPPORTED, str(error)))
            continue
        except Exception as error:
            # One case failing must not cost the other five. The adapter ran in
            # its own environment, so this is usually an API that moved.
            results.append(
                Result.blank(case, FAILED, f"{type(error).__name__}: {error}")
            )
            continue

        try:
            if op.startswith("read"):
                # Reads only. The checksum needs one call to hash, and a read is
                # cheap and side-effect free. A write or a pyramid build is
                # neither, and this used to run for those too -- an extra whole
                # store written per case that nobody asked for and no column
                # reported.
                extra["checksum"] = _checksum(measured.fn())
            seconds, peak = measure(measured.fn, repeats=repeats, warmup=job.warmup)
        except Exception as error:
            results.append(
                Result.blank(case, FAILED, f"{type(error).__name__}: {error}")
            )
            continue

        extra.update(measured.extra)
        target = measured.extra.get("target")
        if target:
            # Every operation that produced a store gets sized, not just the
            # ones named `write_*`: a pyramid is a store too, and "how big is
            # what it wrote" is the check that keeps a fast column honest.
            path = Path(target)
            extra["bytes"] = _stored_bytes(path)
            if audit is not None:
                extra.update(audit(path, spec))
        if getattr(adapter, "NATIVE", False):
            # tracemalloc cannot see this implementation's buffers, and a
            # 0.0 in the peak column would read as 'uses no memory' rather
            # than 'not measurable here'.
            peak = float("nan")
        results.append(Result(case, seconds, peak, measured.note, extra=extra))
    return results
