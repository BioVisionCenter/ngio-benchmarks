"""The child side of a comparison: one implementation, one case.

Runs inside the isolated environment holding exactly that implementation. The
parent has already decided which cases this adapter can express, built any
shared fixture, and written the whole instruction set into the job -- so there
is nothing to negotiate here.

One case per process, not one implementation per process. The extra interpreter
starts buy an honest memory column: `getrusage` reports a high-water mark that
only ever rises, so a child running six cases can only report the largest of
them and attribute it to all six. It also stops a case inheriting the previous
one's heap, free lists and warmed imports, which is the same reason each
implementation gets its own environment rather than sharing one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from ngio_benchmarks.compare import _pipeline
from ngio_benchmarks.core import images as images_mod
from ngio_benchmarks.core.axes import Case
from ngio_benchmarks.core.config import DEFAULT_REPEATS
from ngio_benchmarks.core.measure import (
    FAILED,
    UNSUPPORTED,
    Measured,
    Result,
    Unsupported,
    flush_seconds,
    load_average,
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

    Memory is deliberately *not* here any more. It used to be, read once at the
    end of a child that had run every case for one implementation, which made it
    the maximum over all of them wearing each one's label. It is now recorded by
    `run` around the single case this process exists for.
    """
    package = getattr(adapter, "DISTRIBUTION", None) or getattr(adapter, "NAME", label)
    return {
        "impl": label,
        "impl_version": version(package),
        "zarr": version("zarr"),
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


def _case(job: Job) -> tuple[str, str, dict[str, str], dict[str, object]]:
    """Unpack the job's one case into `(op, image, labels, build kwargs)`.

    The labels are what the CSV and the table call this row: the image, the
    canonical downsampling method, and whatever adapter-specific options the
    config swept. They are the axis columns, so they hold tokens a config file
    could have written, never a repr of a kwargs dict.
    """
    op = job.case["op"]
    image = job.case["image"]
    method = job.case.get("method") or ""
    options = dict(job.case.get("options") or {})

    labels = {"image": image}
    if method:
        labels["method"] = method
    # The parent's labels, not `str(value)`: a closed axis' token is `true`
    # where its value is `True`, and the column has to read back as the token a
    # config could have written.
    labels.update(job.case.get("labels") or {})

    kwargs: dict[str, object] = dict(options)
    if method:
        kwargs["method"] = method
    return op, image, labels, kwargs


def run(
    job: Job,
    impls: dict[str, str],
    audit: Callable[..., dict[str, str]] | None = None,
) -> list[Result]:
    """Measure this job's single case, in the current environment.

    Returns a one-element list because the caller writes a CSV either way and a
    failure still has to produce a row -- a case that vanishes from the table is
    indistinguishable from one nobody selected.

    `audit` inspects whatever the case wrote and returns extra columns for it --
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

    # Before `build`, which is where several adapters already create arrays, and
    # before the adapter imports its library at all. The pipeline is chosen at
    # array-open time from a global, so installing it here reaches every library
    # this child goes on to use without any of them cooperating -- and doing it
    # outside the timing means the toggle is not part of what is measured.
    pipeline = job.case.get("pipeline") or _pipeline.DEFAULT
    _pipeline.install(pipeline)

    op, image, labels, kwargs = _case(job)
    spec = registry[image]
    case = Case(op, labels, {})
    variant = " ".join(f"{k}={v}" for k, v in labels.items() if k != "image")
    extra: dict[str, str] = {
        "image": image,
        "zarr_format": str(spec.zarr_format),
        # Filled on every row, not only the rows of a config that swept it: which
        # stack encoded these bytes is a fact about the measurement, and a column
        # that was blank by default would make the swept rows look like the only
        # ones with a pipeline. `variant` is where "what this file varied" lives.
        "pipeline": pipeline,
        "pipeline_version": version(_pipeline.distribution(pipeline)),
        # Empty where nothing varies, which is every row of `compare-io`. The
        # report drops the column entirely in that case rather than printing a
        # word like `default` that implies a choice was available.
        "variant": variant,
        "loadavg": f"{load_average():.2f}",
    }

    def failed(error: Exception) -> list[Result]:
        return [
            Result.blank(case, FAILED, f"{type(error).__name__}: {error}", dict(extra))
        ]

    try:
        measured: Measured = adapter.build(op, spec, root, **kwargs)
    except Unsupported as error:
        return [Result.blank(case, UNSUPPORTED, str(error), dict(extra))]
    except Exception as error:
        # The adapter ran in its own environment, so this is usually an API
        # that moved. It is one row, and the other cases are other processes.
        return failed(error)

    # Taken here on purpose: after this environment's imports and after the
    # adapter has materialised whatever input it needs, but before anything is
    # measured. Subtracted from `proc_peak_mb` it is the case's own cost; on its
    # own it is the price of admission for this library, which for several of
    # them is the larger number.
    extra["rss_base_mb"] = f"{process_peak_mb():.1f}"

    try:
        if op.startswith("read"):
            # Reads only. The checksum needs one call to hash, and a read is
            # cheap and side-effect free. A write or a pyramid build is
            # neither, and this used to run for those too -- an extra whole
            # store written per case that nobody asked for and no column
            # reported.
            extra["checksum"] = _checksum(measured.fn())
        timing = measure(
            measured.fn,
            repeats=repeats,
            warmup=job.warmup,
            setup=measured.setup,
        )
    except Exception as error:
        return failed(error)

    extra["sync_seconds"] = f"{flush_seconds():.6f}"
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
    if getattr(adapter, "NATIVE", False) or pipeline in _pipeline.NATIVE:
        # tracemalloc cannot see this implementation's buffers, and a
        # 0.0 in the peak column would read as 'uses no memory' rather
        # than 'not measurable here'. True of a pure-Python library too
        # once its chunks are encoded in Rust: the pipeline moves the
        # buffers out of tracemalloc's reach, whoever called it.
        timing = timing._replace(peak_mb=float("nan"))
    # Last, so it covers everything this process did.
    extra["proc_peak_mb"] = f"{process_peak_mb():.1f}"
    return [Result.timed(case, timing, measured.note, extra)]
