"""The in-environment worker: `python -m ngio_benchmarks._child <job.json>`.

Invoked by `uv run --no-project` with the project root as the working
directory, so this package is importable from `sys.path[0]` without being
installed -- which is what lets a child's environment hold exactly the library
under test and nothing else.

It is told one environment or one implementation and never reads a config file,
so there is nothing here that could spawn another child.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ngio_benchmarks.core.job import Job
from ngio_benchmarks.core.output import as_row, write_csv


def _internal(job: Job) -> int:
    """Measure the internal blocks against whatever ngio this environment has."""
    from ngio_benchmarks.internal._run import (
        SCHEMA,
        environment,
        load,
        resolve,
        run_blocks,
        unavailable_results,
    )

    modules, missing = load(list(job.blocks))
    resolved = resolve(modules, job.overrides)
    results, _ = run_blocks(
        modules,
        resolved,
        Path(job.root),
        repeats=job.repeats,
        warmup=job.warmup,
        quiet=True,
    )
    results += unavailable_results(missing)
    env = environment(job.label)
    write_csv(Path(job.out), SCHEMA, [as_row(r, SCHEMA, env) for r in results])
    return 0


def _compare(job: Job) -> int:
    """Measure one implementation's cases in its own environment."""
    import importlib

    from ngio_benchmarks.compare._run import environment, run

    suite = importlib.import_module(
        "ngio_benchmarks.compare.io"
        if job.suite == "compare-io"
        else "ngio_benchmarks.compare.create"
    )
    adapter = importlib.import_module(suite.IMPLS[job.impl])
    results = run(job, suite.IMPLS, getattr(suite, "audit", None))
    # `job.impl`, not `job.label`: the label is a progress line naming the whole
    # case, while the `impl` column is the implementation's name and nothing
    # else -- it is what the table groups and the CSV pivots on.
    env = environment(adapter, job.impl)
    write_csv(
        Path(job.out),
        suite.SCHEMA,
        [as_row(r, suite.SCHEMA, env) for r in results],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one job."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: python -m ngio_benchmarks._child <job.json>")
    job = Job.read(Path(arguments[0]))
    if job.suite == "internal":
        return _internal(job)
    return _compare(job)


if __name__ == "__main__":
    raise SystemExit(main())
