"""The runner both comparison suites share.

`compare-io` and `compare-create` differ only in their registry of adapters and
their list of operations, so they are one runner parameterised by a `Suite`.
`compare-create` has a single operation and could have dropped the axis
entirely; keeping it means the two suites produce the same shape of table and
the same code decides what is unsupported.

The parent never imports a peer library. It imports each adapter *module* to
read its declarations, works out which cases that adapter can express, and then
hands the survivors to a child running inside the adapter's own environment.
Cases the adapter excludes never reach a child at all -- they are rendered as
`unsupported` by the parent, which is why a comparison table can be printed
even when nothing installs.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, NamedTuple

from ngio_benchmarks.core import cli as cli_core
from ngio_benchmarks.core import config as config_core
from ngio_benchmarks.core import envs as envs_core
from ngio_benchmarks.core import images as images_mod
from ngio_benchmarks.core.axes import Case
from ngio_benchmarks.core.config import DEFAULT_REPEATS
from ngio_benchmarks.core.job import Job
from ngio_benchmarks.core.measure import UNSUPPORTED, Result, executions
from ngio_benchmarks.core.output import (
    Schema,
    as_row,
    check_csv,
    report_matrix,
    write_csv,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec


class Suite(NamedTuple):
    """One comparison suite: what it measures and who is in the running."""

    name: str
    prog: str
    description: str
    #: implementation name -> adapter module path
    impls: dict[str, str]
    ops: tuple[str, ...]
    schema: Schema
    #: The image specs swept when a config names none.
    default_images: tuple[str, ...]
    #: Builds whatever the children share -- the source pixels, and for the io
    #: suite the read fixture. Run in the parent, before any environment
    #: installs, so no implementation has to carry a library purely for setup
    #: and no child can leave a half-written store behind for the next one.
    prepare: Callable[[Path, list[ImageSpec], list[str]], None] | None = None


def _keys(suite: Suite) -> tuple[str, ...]:
    """Every config key this suite accepts.

    `[axes]` is absent on purpose. The internal suite's axes belong to
    individual blocks and need qualifying; here the three axes are the suite's
    own, so they are plain top-level lists named exactly like the CSV columns
    they land in -- `impl`, `op`, `image`.

    The singular/plural split is load-bearing and runs through the whole config
    surface: a **singular** key is an axis and holds a list of labels, a
    **plural** table defines what those labels mean. `image` selects,
    `[images.<name>]` defines; `impl` selects, `[env.<impl>]` pins. Without it
    the selection list and its definition table collide on one name.
    """
    common = tuple(k for k in config_core.COMMON_KEYS if k != "axes")
    return (*common, "impl", "op", "image", "env")


def load_adapters(suite: Suite, selected: list[str]) -> dict[str, Any]:
    """Import each selected adapter module for its declarations only.

    An adapter that raises here is a bug in this repo, not a missing library:
    the whole point of the protocol is that the module imports without the
    library it wraps.
    """
    return {name: importlib.import_module(suite.impls[name]) for name in selected}


def plan(
    suite: Suite,
    adapters: dict[str, Any],
    ops: list[str],
    specs: list[ImageSpec],
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, Result]]]:
    """Split the full product into what each adapter will run and what it cannot.

    Returns `{impl: [(op, image), ...]}` and the blank results for the rest.
    """
    runnable: dict[str, list[tuple[str, str]]] = {}
    excluded: list[tuple[str, Result]] = []
    for name, adapter in adapters.items():
        supports = set(getattr(adapter, "SUPPORTS", ()))
        formats = set(getattr(adapter, "FORMATS", (2, 3)))
        pairs = []
        for op in ops:
            for spec in specs:
                # The operation is the case's *group*, not part of its label:
                # a row must be the same case in every column, so the label
                # holds only the axes that vary within one table.
                case = Case(op, {"image": spec.name}, {})
                if op not in supports:
                    excluded.append(
                        (name, Result.blank(case, UNSUPPORTED, f"{name} has no {op}"))
                    )
                elif spec.zarr_format not in formats:
                    excluded.append(
                        (
                            name,
                            Result.blank(
                                case,
                                UNSUPPORTED,
                                f"{name} cannot use zarr v{spec.zarr_format}",
                            ),
                        )
                    )
                else:
                    pairs.append((op, spec.name))
        runnable[name] = pairs
    return runnable, excluded


def main(suite: Suite, argv: list[str] | None = None) -> int:
    """Run one comparison suite from a config file."""
    args = cli_core.parse(suite.prog, suite.description, argv)
    config: Path = args.config
    data = config_core.read(config, _keys(suite))
    settings = config_core.common(data, config, ())

    registry = images_mod.registry(settings.images)
    impls = config_core.names(data, "impl", suite.impls, config) or list(suite.impls)
    ops = config_core.names(data, "op", suite.ops, config) or list(suite.ops)
    chosen = config_core.names(data, "image", registry, config) or list(
        suite.default_images
    )
    specs = [registry[name] for name in chosen]

    check_envs(data, impls, config)
    adapters = load_adapters(suite, impls)
    runnable, excluded = plan(suite, adapters, ops, specs)

    if args.list:
        _describe(suite, adapters, runnable, excluded, specs, config, settings)
        return 0

    if settings.csv:
        check_csv(settings.csv, suite.schema)

    envs_core.require_uv()
    # Keyed by implementation and emitted in the order the config named them,
    # so the table's columns read in that order too -- and so the ratio column
    # is anchored on the first implementation asked for rather than on
    # whichever one happened to contribute the first row.
    by_impl: dict[str, list[dict[str, str]]] = {name: [] for name in impls}
    for name, result in excluded:
        by_impl[name].append(as_row(result, suite.schema, {"impl": name}))

    with cli_core.data_root(settings.keep) as root:
        if suite.prepare is not None:
            live = sorted({op for pairs in runnable.values() for op, _ in pairs})
            used = [
                s
                for s in specs
                if any(s.name == i for p in runnable.values() for _, i in p)
            ]
            print("preparing fixtures ...", flush=True)
            suite.prepare(root, used, live)

        for name, pairs in runnable.items():
            if not pairs:
                continue
            adapter = adapters[name]
            job = Job(
                suite=suite.name,
                label=name,
                root=str(root),
                repeats=settings.repeats,
                warmup=settings.warmup,
                ops=tuple(f"{op}\t{image}" for op, image in pairs),
                impl=name,
                # Only the config's additions travel; the built-in registry is
                # the same code on both sides.
                images={
                    spec.name: spec
                    for spec in specs
                    if spec.name in settings.images
                    or spec.name in {i for _, i in pairs}
                },
            )
            child, error = envs_core.run_child(
                name,
                job,
                _with_args(adapter, data, name, config),
                python=_env_table(data, name).get("python")
                or getattr(adapter, "PYTHON", None),
                quiet=settings.quiet,
            )
            if error:
                by_impl[name].append(_failure_row(suite, name, error))
                continue
            by_impl[name] += [{**row, "impl": name} for row in child]

    rows = [row for name in impls for row in by_impl[name]]
    if settings.csv:
        write_csv(settings.csv, suite.schema, rows)
        print(f"wrote {settings.csv}")
    report_matrix(rows, suite.schema)
    return 0


def check_envs(data: dict[str, Any], impls: list[str], config: Path) -> None:
    """Validate every `[env.<impl>]` table, before anything installs.

    Eager, and run on the `--list` path too. A typo'd key here would otherwise
    fall back to the adapter's default pins and be discovered only after six
    environments had been built -- and the run would have measured versions
    nobody asked for while looking like it worked.
    """
    envs = data.get("env", {})
    if not isinstance(envs, dict):
        raise SystemExit(f"{config}: `env` must be a table, e.g. [env.tensorstore]")
    for name, table in envs.items():
        if name not in impls:
            raise SystemExit(
                f"{config}: [env.{name}] names an implementation that is not "
                f"selected; selected: {', '.join(impls)}"
            )
        if not isinstance(table, dict):
            raise SystemExit(f"{config}: [env.{name}] must be a table")
        unknown = sorted(set(table) - {"requires", "python"})
        if unknown:
            raise SystemExit(
                f"{config}: [env.{name}] has unknown key {unknown[0]!r}; "
                "valid keys are requires, python"
            )
        if "requires" in table and not (
            isinstance(table["requires"], list) and table["requires"]
        ):
            raise SystemExit(
                f"{config}: [env.{name}].requires must be a non-empty list of "
                "requirements"
            )


def _env_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    """The `[env.<impl>]` table for one implementation. Validated already."""
    return data.get("env", {}).get(name, {})


def _with_args(
    adapter: Any, data: dict[str, Any], name: str, config: Path
) -> list[str]:
    """The uv arguments for one implementation's environment.

    An `[env.<impl>] requires = [...]` table replaces the adapter's default
    pins entirely. That is what lets a committed config name the exact peer
    version its numbers came from, rather than "whatever resolved the day it
    ran" -- which is the difference between a result and an anecdote.
    """
    requires = _env_table(data, name).get("requires") or list(
        getattr(adapter, "REQUIRES", ())
    )
    args: list[str] = []
    for token in requires:
        args += envs_core.requirement(token, config.parent)
    return args


def _failure_row(suite: Suite, name: str, error: str) -> dict[str, str]:
    """One row recording that an implementation never ran."""
    row = dict.fromkeys(suite.schema.fields, "")
    row.update(
        {
            "impl": name,
            # Group `-` so the ratio in the comparison table is never taken
            # against a column that produced nothing.
            suite.schema.group: "-",
            "case": "-",
            "status": "unavailable",
            "note": f"environment failed ({error})",
        }
    )
    return row


def _describe(
    suite: Suite,
    adapters: dict[str, Any],
    runnable: dict[str, list[tuple[str, str]]],
    excluded: list[tuple[str, Result]],
    specs: list[ImageSpec],
    config: Path,
    settings: Any,
) -> None:
    """Print what this config would install and measure, then stop.

    Worth its own flag rather than a config key: with one environment per
    implementation to install before the first measurement, a config that
    selects nothing is expensive to discover any later than this.
    """
    print(f"\n{suite.name}  ({config})")
    print("\nimages:")
    for spec in specs:
        shards = f" shards={spec.shards}" if spec.shards else ""
        print(
            f"  {spec.name:<14} {spec.shape} chunks={spec.chunks}{shards} "
            f"{spec.dtype} {spec.compressors} zarr v{spec.zarr_format} "
            f"levels={spec.levels}  ({spec.nbytes / 1024 / 1024:.0f} MB)"
        )

    print("\nimplementations:")
    width = max(len(n) for n in adapters)
    for name, adapter in adapters.items():
        pairs = runnable[name]
        requires = " ".join(getattr(adapter, "REQUIRES", ()))
        pin = getattr(adapter, "PYTHON", None)
        python = f" [python {pin}]" if pin else ""
        repeats = settings.repeats or getattr(adapter, "REPEATS", DEFAULT_REPEATS)
        runs = executions(repeats, settings.warmup)
        print(
            f"  {name:<{width}}  {len(pairs):>3} cases x {runs} runs{python}  "
            f"{requires}"
        )

    if excluded:
        print("\nunsupported:")
        for name, result in excluded:
            label = f"{name} {result.case.block} {result.case.label}"
            print(f"  {label:<52} {result.note}")

    total = sum(len(p) for p in runnable.values())
    # `runs` above, not just `repeats`: the callable is executed warmup times,
    # then `repeats` times for the timing, then once or twice more with
    # `tracemalloc` on. For a pyramid build that is the difference between one
    # store written and four, and it should be visible before the run, not
    # inferred from how long it took.
    print(
        f"\n{len([n for n, p in runnable.items() if p])} environments, "
        f"{total} cases, {settings.repeats or 'per-adapter'} repeats, "
        f"{settings.warmup} warmup"
    )
    if settings.csv:
        print(f"csv: {settings.csv}")
    print()
