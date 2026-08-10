"""`ngio-bench-internal <config.toml> [--list]`.

Two forms, chosen by whether the config declares `[[environments]]`:

* none -- measure in this interpreter, against whatever ngio is installed
* one or more -- spawn a `uv` environment per entry and merge their rows

The child is invoked with a job naming exactly one environment, which is also
what stops this recursing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.core import cli as cli_core
from ngio_benchmarks.core import config as config_core
from ngio_benchmarks.core import envs as envs_core
from ngio_benchmarks.core.job import Job
from ngio_benchmarks.core.output import (
    as_row,
    check_csv,
    report,
    report_matrix,
    write_csv,
)
from ngio_benchmarks.internal import BLOCKS
from ngio_benchmarks.internal._run import (
    SCHEMA,
    environment,
    load,
    print_list,
    resolve,
    run_blocks,
    unavailable_results,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.config import Common

KEYS = (*config_core.COMMON_KEYS, "blocks")


def _describe(settings: Common, blocks: list[str], config: Path) -> None:
    """Print the environments a run would install, for `--list`."""
    if not settings.environments:
        print("\nenvironments: none — measuring in the current interpreter")
        return
    print("\nenvironments:")
    for env in settings.environments:
        args = envs_core.uv_args(env, config.parent)
        print(f"  {env.name:<24} uv run {' '.join(args)}")
    print(f"\n{len(settings.environments)} environments x {len(blocks)} blocks")


def main(argv: list[str] | None = None) -> int:
    """Run the internal suite from a config file."""
    args = cli_core.parse("ngio-bench-internal", "measure ngio against itself", argv)
    config: Path = args.config
    data = config_core.read(config, KEYS)
    settings = config_core.common(data, config, BLOCKS)
    blocks = config_core.names(data, "blocks", BLOCKS, config) or sorted(BLOCKS)

    modules, missing = load(blocks)
    resolved = resolve(modules, settings.overrides)

    if args.list:
        print_list(modules, resolved, missing, settings.repeats, settings.warmup)
        _describe(settings, blocks, config)
        if settings.csv:
            print(f"csv: {settings.csv}")
        print()
        return 0

    if settings.csv:
        # Once, before anything runs: an unusable path should cost nothing
        # rather than being discovered after the environments have installed.
        check_csv(settings.csv, SCHEMA)

    if settings.environments:
        return _run_environments(settings, blocks, config)
    return _run_here(settings, modules, resolved, missing)


def _run_here(settings, modules, resolved, missing) -> int:
    """Measure in the current interpreter."""
    with cli_core.data_root(settings.keep) as root:
        results, skipped = run_blocks(
            modules,
            resolved,
            root,
            repeats=settings.repeats,
            warmup=settings.warmup,
            quiet=settings.quiet,
        )
        results += unavailable_results(missing)
        if settings.csv:
            env = environment("current")
            write_csv(settings.csv, SCHEMA, [as_row(r, SCHEMA, env) for r in results])
            print(f"wrote {settings.csv}")
        if not settings.quiet:
            report(results)
            if skipped:
                # Never let a bounded sweep read as a complete one.
                print(f"({skipped} cases skipped: not measured at that size)\n")
    return 0


def _run_environments(settings, blocks: list[str], config: Path) -> int:
    """Spawn one `uv` environment per entry and merge their rows.

    `keep` is deliberately not honoured across environments: a fixture written
    by ngio 1.0.0 and read back by the working tree is a different experiment
    from the one that was asked for, so every child gets its own temp root.
    """
    envs_core.require_uv()
    if settings.keep:
        print("note: `keep` does not apply across environments; ignoring it")

    rows: list[dict[str, str]] = []
    for env in settings.environments:
        with cli_core.data_root(keep=False) as root:
            job = Job(
                suite="internal",
                label=env.name,
                root=str(root),
                repeats=settings.repeats,
                warmup=settings.warmup,
                blocks=tuple(blocks),
                overrides=settings.overrides,
                # Only the config's additions travel; the built-in registry is
                # the same code on both sides.
                images=settings.images,
            )
            child, error = envs_core.run_child(
                env.name,
                job,
                envs_core.uv_args(env, config.parent),
                python=env.python,
                quiet=True,
            )
            if error:
                rows.append(_failure_row(env.name, error))
                continue
            # The child does not know which environment it is; the parent does.
            rows += [{**row, "env": env.name} for row in child]

    if settings.csv:
        write_csv(settings.csv, SCHEMA, rows)
        print(f"wrote {settings.csv}")
    report_matrix(rows, SCHEMA)
    return 0


def _failure_row(label: str, error: str) -> dict[str, str]:
    """One row recording that an environment never ran.

    A failed environment still gets a row, with block `-` so the ratio in the
    comparison table is not taken against a column that produced nothing.
    """
    row = dict.fromkeys(SCHEMA.fields, "")
    row.update({"env": label, "block": "-", "case": "-", "status": "unavailable"})
    row["note"] = f"environment failed ({error})"
    return row
