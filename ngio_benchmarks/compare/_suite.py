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

from ngio_benchmarks.compare import _pipeline
from ngio_benchmarks.core import cli as cli_core
from ngio_benchmarks.core import config as config_core
from ngio_benchmarks.core import envs as envs_core
from ngio_benchmarks.core import images as images_mod
from ngio_benchmarks.core.axes import Case, apply_overrides, cases
from ngio_benchmarks.core.config import DEFAULT_REPEATS
from ngio_benchmarks.core.job import Job
from ngio_benchmarks.core.measure import UNSUPPORTED, Result, executions
from ngio_benchmarks.core.output import (
    Schema,
    as_row,
    check_csv,
    log,
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
    #: The codec pipelines this suite sweeps along, or empty for a suite that
    #: has no such axis. `compare-io` leaves it empty because it expresses the
    #: same question as a pair of implementations instead -- see `_pipeline`.
    pipelines: tuple[str, ...] = ()
    #: Builds whatever the children share -- the source pixels, and for the io
    #: suite the read fixture. Run in the parent, before any environment
    #: installs, so no implementation has to carry a library purely for setup
    #: and no child can leave a half-written store behind for the next one.
    prepare: Callable[[Path, list[ImageSpec], list[str]], None] | None = None


def _keys(suite: Suite) -> tuple[str, ...]:
    """Every config key this suite accepts.

    `[axes]` is absent on purpose. The internal suite's axes belong to
    individual blocks and need qualifying; here the axes are the suite's own, so
    they are plain top-level lists named exactly like the CSV columns they land
    in -- `impl`, `op`, `image`, `method`.

    The singular/plural split is load-bearing and runs through the whole config
    surface: a **singular** key is an axis and holds a list of labels, a
    **plural** table defines what those labels mean. `image` selects,
    `[images.<name>]` defines; `impl` selects, `[env.<impl>]` pins; `method`
    selects, `[options.<impl>]` says what else that one writer can be asked.
    Without it the selection list and its definition table collide on one name.

    `pipeline` is offered only by a suite that declares one, so `compare-io`
    rejects it as an unknown key rather than accepting a setting it would then
    ignore -- which is the same reason the suites have separate console scripts.
    """
    common = tuple(k for k in config_core.COMMON_KEYS if k != "axes")
    pipeline = ("pipeline",) if suite.pipelines else ()
    return (*common, "impl", "op", "image", "method", *pipeline, "env", "options")


#: The name an `[options.<impl>]` table uses to bypass the canonical `method`
#: axis and name one library's own downsampling filter directly.
NATIVE = "method_native"


def methods(adapters: dict[str, Any]) -> list[str]:
    """The canonical downsampling vocabulary, across the selected adapters.

    The union of what they declare, so a config can ask for `gaussian` even
    though only one writer has one -- the rest report `unsupported`, which is
    the answer, and a silently narrower matrix would not have been.
    """
    seen: list[str] = []
    for adapter in adapters.values():
        for name in getattr(adapter, "METHODS", {}):
            if name not in seen:
                seen.append(name)
    return seen


def _takes_native(adapter: Any) -> bool:
    """Whether this adapter's library names its downsampling filter at all.

    `METHOD_NATIVE = False` on an adapter that translates the canonical names
    into something other than a filter argument, so there is nothing for a
    config to pass through.
    """
    return bool(getattr(adapter, "METHODS", {})) and getattr(
        adapter, "METHOD_NATIVE", True
    )


def check_options(data: dict[str, Any], adapters: dict[str, Any], config: Path) -> None:
    """Validate every `[options.<impl>]` table, before anything installs.

    Eager and total, like `check_envs` beside it. An `[options]` table is the
    one place a config can multiply the size of a run, so a typo here is
    expensive twice over: it would install six environments and then measure a
    matrix nobody asked for.
    """
    tables = data.get("options", {})
    if not isinstance(tables, dict):
        raise SystemExit(f"{config}: `options` must be a table, e.g. [options.ngio]")
    for name, table in tables.items():
        if name not in adapters:
            raise SystemExit(
                f"{config}: [options.{name}] names an implementation that is not "
                f"selected; selected: {', '.join(adapters)}"
            )
        if not isinstance(table, dict):
            raise SystemExit(f"{config}: [options.{name}] must be a table")
        adapter = adapters[name]
        declared = getattr(adapter, "OPTIONS", {})
        # `method_native` is offered only by an adapter whose library takes a
        # filter by name. ngio takes a mode and an interpolation order, and
        # bioio takes nothing at all, so both decline it -- and declining it
        # here rather than raising at build time means `--list` says so before
        # six environments install.
        valid = (*declared, *((NATIVE,) if _takes_native(adapter) else ()))
        for option, values in table.items():
            if option not in valid:
                raise SystemExit(
                    f"{config}: [options.{name}] has no {option!r}; "
                    f"{name} takes: {', '.join(valid) or 'no options'}"
                )
            config_core.tokens(f"options.{name}.{option}", values)
        if NATIVE in table and data.get("method"):
            # Both would be asking the same question in two vocabularies, and
            # the row could only be labelled with one of them.
            raise SystemExit(
                f"{config}: [options.{name}] sets `{NATIVE}` while `method` is "
                "also set. They are two spellings of one axis -- `method` is the "
                f"canonical vocabulary every writer translates, `{NATIVE}` names "
                f"{name}'s own filter directly. Use one or the other."
            )


def _option_cases(
    adapter: Any, table: dict[str, Any], impl: str
) -> list[tuple[dict[str, Any], dict[str, str]]]:
    """Every combination of adapter-specific options one impl should sweep.

    Returns `(values, labels)` pairs: the values are what `build` is called
    with, the labels are the tokens the config wrote and the table and CSV will
    show. They differ whenever an option is a closed axis -- `use_tensorstore`
    is the token `true` and the value `True` -- and the rule that a CSV cell and
    a config token are the same token is worth the extra tuple.

    Only options the config actually named enter the product. An adapter's
    `OPTIONS` is a declaration of what *can* be swept, not a default sweep: left
    alone, every option keeps whatever `method` implied, and the matrix is the
    size it was before anyone asked for more.

    The open/closed axis rules and their override semantics are `core.axes`'
    job, not this module's. `method_native` is handled outside that, because its
    vocabulary is whatever the wrapped library accepts and this suite has no
    business enumerating it.
    """
    native = table.get(NATIVE)
    overrides = tuple(
        (impl, option, config_core.tokens(f"options.{impl}.{option}", values))
        for option, values in table.items()
        if option != NATIVE
    )
    if not overrides:
        combinations: list[tuple[dict[str, Any], dict[str, str]]] = [({}, {})]
    else:
        declared = getattr(adapter, "OPTIONS", {})
        asked = {name for _, name, _ in overrides}
        resolved = apply_overrides(
            impl, {k: v for k, v in declared.items() if k in asked}, overrides
        )
        combinations = [
            (dict(case.values), dict(case.labels)) for case in cases(impl, resolved)
        ]
    if not native:
        return combinations
    return [
        ({**values, NATIVE: str(value)}, {**labels, NATIVE: str(value)})
        for value in native
        for values, labels in combinations
    ]


def load_adapters(suite: Suite, selected: list[str]) -> dict[str, Any]:
    """Import each selected adapter module for its declarations only.

    An adapter that raises here is a bug in this repo, not a missing library:
    the whole point of the protocol is that the module imports without the
    library it wraps.
    """
    return {name: importlib.import_module(suite.impls[name]) for name in selected}


def _variant(case: dict[str, Any]) -> str:
    """The part of a case's identity that is not the image: method and options.

    `method=mean mode=numpy`, or empty when the config swept nothing. It is what
    names a row in the table, so it holds config tokens and nothing else.
    """
    return " ".join(f"{k}={v}" for k, v in _labels(case).items() if k != "image")


def _labels(case: dict[str, Any]) -> dict[str, str]:
    """The axis labels of one case, image included. Matches `compare._run`."""
    labels = {"image": case["image"]}
    if case.get("method"):
        labels["method"] = str(case["method"])
    labels.update(case.get("labels") or {})
    return labels


def plan(
    suite: Suite,
    adapters: dict[str, Any],
    ops: list[str],
    specs: list[ImageSpec],
    wanted: list[str] | None = None,
    options: dict[str, dict[str, Any]] | None = None,
    pipelines: list[str] | None = None,
    *,
    label_pipeline: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, Result]]]:
    """Split the full product into what each adapter will run and what it cannot.

    Returns `{impl: [case, ...]}` and the blank results for the rest, where a
    case is the resolved `{"op", "image", "method", "pipeline", "options"}` a
    child is handed verbatim.

    `wanted` is the canonical `method` axis. An adapter that does not declare a
    translation for one of its values is excluded from that row rather than
    quietly given its own default -- a writer that answered a request for
    `nearest` with a gaussian would be the fastest kind of wrong.

    `pipelines` is the codec pipeline axis, which is nobody's option: it is
    installed by the child before the adapter is asked for anything, so every
    library that writes through zarr-python takes it without cooperating. Only
    the adapters that do not -- a streaming C++ writer, one that picks its own
    zarr stack -- declare themselves out of it.

    `label_pipeline` is whether the pipeline joins the row's variant label. It
    does when the config named the axis and not otherwise, which is the rule
    `_option_cases` already follows: a table names what the file asked to vary,
    and one value nobody chose between is not a variant.
    """
    runnable: dict[str, list[dict[str, Any]]] = {}
    excluded: list[tuple[str, Result]] = []
    for name, adapter in adapters.items():
        supports = set(getattr(adapter, "SUPPORTS", ()))
        formats = set(getattr(adapter, "FORMATS", (2, 3)))
        translations = getattr(adapter, "METHODS", {})
        # An empty `wanted` means the suite has no method axis at all (compare-io
        # does not); one entry that the adapter cannot translate is a real gap.
        asked = wanted or [""]
        stacks = pipelines or [_pipeline.DEFAULT]
        variants = _option_cases(adapter, (options or {}).get(name, {}), name)
        planned: list[dict[str, Any]] = []
        for op in ops:
            for spec in specs:
                for method in asked:
                    for pipeline in stacks:
                        for values, labels in variants:
                            # `method_native` names the library's own filter, so
                            # the canonical column is blank rather than guessed.
                            entry = {
                                "op": op,
                                "image": spec.name,
                                "method": "" if NATIVE in values else method,
                                "pipeline": pipeline,
                                "options": dict(values),
                                # Pipeline before the options, so a variant
                                # reads suite axes first and one library's own
                                # settings after -- `method=mean pipeline=zarrs
                                # mode=numpy`.
                                "labels": (
                                    {"pipeline": pipeline} if label_pipeline else {}
                                )
                                | dict(labels),
                            }
                            reason = _excluded(
                                adapter, name, op, spec, entry["method"], pipeline,
                                values, supports, formats, translations,
                            )  # fmt: skip
                            if not reason:
                                planned.append(entry)
                                continue
                            case = Case(op, _labels(entry), {})
                            excluded.append(
                                (name, Result.blank(case, UNSUPPORTED, reason))
                            )
        runnable[name] = planned
    return runnable, excluded


def _excluded(
    adapter: Any,
    name: str,
    op: str,
    spec: ImageSpec,
    method: str,
    pipeline: str,
    values: dict[str, Any],
    supports: set[str],
    formats: set[int],
    translations: dict[str, Any],
) -> str:
    """Why this adapter cannot run this case, or an empty string if it can.

    Five distinct gaps, kept distinct: an operation it has no API for, a zarr
    format it cannot read or write, a downsampling filter it has no equivalent
    of, a codec pipeline that is not in its write path at all, and one that
    cannot encode this image's zarr format. In a comparison table they are
    different facts about the library, and collapsing them into one blank cell
    loses the interesting one.
    """
    if op not in supports:
        return f"{name} has no {op}"
    if spec.zarr_format not in formats:
        return f"{name} cannot use zarr v{spec.zarr_format}"
    if method and translations and method not in translations:
        return (
            f"{name} has no {method} downsampling "
            f"(it offers: {', '.join(translations)})"
        )
    return _excluded_pipeline(adapter, name, spec, pipeline, values)


def _excluded_pipeline(
    adapter: Any, name: str, spec: ImageSpec, pipeline: str, values: dict[str, Any]
) -> str:
    """Why this codec pipeline cannot apply to this case, or an empty string.

    An adapter declares itself out of the axis with `PIPELINES`, and judges a
    combination only it can judge -- ngff-zarr's tensorstore backend, which
    never reaches zarr-python's pipeline -- with `excludes_pipeline`. Both are
    answered here rather than by an `Unsupported` at build time so that `--list`
    shows the gap and no child is started to discover it.

    The hook is asked first, so an adapter whose reason is not the generic one
    can say its own. iohub's is not "it does not go through zarr-python" -- it
    does; it is that iohub chooses its zarr stack through its own option, and a
    cell that named the wrong reason is the thing these messages exist to avoid.
    """
    hook = getattr(adapter, "excludes_pipeline", None)
    if hook is not None and (reason := hook(pipeline, values)):
        return reason
    declared = getattr(adapter, "PIPELINES", None)
    if declared is not None and pipeline not in declared:
        return f"{name} does not encode through zarr-python's codec pipeline"
    return _pipeline.unsupported(pipeline, spec.zarr_format)


def main(suite: Suite, argv: list[str] | None = None) -> int:
    """Run one comparison suite from a config file."""
    args = cli_core.parse(suite.prog, suite.description, argv)
    config: Path = args.config
    data = config_core.read(config, _keys(suite))
    settings = config_core.common(data, config, ())
    if settings.environments:
        raise SystemExit(
            f"{config}: `[[environments]]` has no effect in {suite.name} -- "
            "this suite pins one ngio per run via `[env.ngio] requires = "
            '[...]`, not a matrix. Run the config once per build and let the '
            "rows land in the same `csv`."
        )

    registry = images_mod.registry(settings.images)
    impls = config_core.names(data, "impl", suite.impls, config) or list(suite.impls)
    ops = config_core.names(data, "op", suite.ops, config) or list(suite.ops)
    chosen = config_core.names(data, "image", registry, config) or list(
        suite.default_images
    )
    specs = [registry[name] for name in chosen]

    check_envs(data, impls, config)
    adapters = load_adapters(suite, impls)
    check_options(data, adapters, config)
    available = methods(adapters)
    wanted = config_core.names(data, "method", available, config) or (
        # `default` when the adapters have a method axis and the config said
        # nothing: measuring what each library does untold is the question this
        # suite started from, and it stays the question by default.
        ["default"] if available else []
    )
    # Not swept by default, on the same reasoning as an `[options]` table: this
    # is the one axis that multiplies the size of every column at once, and a
    # config that did not ask for it should install and measure what it did
    # before the axis existed. Naming it is also what puts it in the row labels.
    pipelines = config_core.names(data, "pipeline", suite.pipelines, config)
    runnable, excluded = plan(
        suite,
        adapters,
        ops,
        specs,
        wanted,
        data.get("options", {}),
        pipelines,
        label_pipeline=pipelines is not None,
    )

    if args.list:
        _describe(suite, adapters, runnable, excluded, specs, config, settings)
        return 0

    if settings.csv:
        check_csv(settings.csv, suite.schema)

    envs_core.require_uv()
    # Keyed by implementation and emitted in the order the config named them,
    # so the table's rows read in that order too -- and so the ratio column is
    # anchored on the first implementation asked for rather than on whichever
    # one happened to contribute the first row.
    by_impl: dict[str, list[dict[str, str]]] = {name: [] for name in impls}
    for name, result in excluded:
        # The variant travels on an excluded row too. A table that named the
        # variant only on the rows that ran would leave `unsupported` looking
        # like a property of the implementation rather than of the one
        # combination it cannot express.
        variant = " ".join(
            f"{k}={v}" for k, v in result.case.labels.items() if k != "image"
        )
        env = {"impl": name, "variant": variant}
        if suite.pipelines:
            # Filled even on a row that never ran, and even when the pipeline is
            # not part of the variant: the column says which stack the row is
            # about, and `unsupported` with an empty one would read as a
            # property of the library rather than of the pairing.
            env["pipeline"] = result.case.labels.get("pipeline", _pipeline.DEFAULT)
        by_impl[name].append(as_row(result, suite.schema, env))

    with cli_core.data_root(settings.keep) as root:
        if suite.prepare is not None:
            live = sorted({c["op"] for cases in runnable.values() for c in cases})
            used = [
                s
                for s in specs
                if any(s.name == c["image"] for cs in runnable.values() for c in cs)
            ]
            log("preparing fixtures ...")
            suite.prepare(root, used, live)

        total_cases = sum(len(planned) for planned in runnable.values())
        index = 0
        for name, planned in runnable.items():
            if not planned:
                continue
            adapter = adapters[name]
            with_args = _with_args(adapter, data, name, config)
            python = _env_table(data, name).get("python") or getattr(
                adapter, "PYTHON", None
            )
            for case in planned:
                index += 1
                # One child per case. `run_child` reuses uv's environment cache,
                # so this pays an interpreter start rather than an install --
                # and it is what makes the memory columns per case rather than
                # per implementation. See `compare._run`.
                # The label is what the progress line says while this child
                # runs. With one child per case there are now dozens of them,
                # so it names the case and not just the implementation.
                detail = " ".join(
                    part for part in (case["op"], case["image"], _variant(case)) if part
                )
                job = Job(
                    suite=suite.name,
                    label=f"{name}  {detail}",
                    root=str(root),
                    repeats=settings.repeats,
                    warmup=settings.warmup,
                    case=case,
                    impl=name,
                    # Only the config's additions travel; the built-in registry
                    # is the same code on both sides.
                    images={
                        spec.name: spec
                        for spec in specs
                        if spec.name in settings.images or spec.name == case["image"]
                    },
                )
                child, error = envs_core.run_child(
                    job.label,
                    job,
                    with_args + _pipeline_args(case, config),
                    python=python,
                    quiet=settings.quiet,
                    index=index,
                    total=total_cases,
                )
                if error:
                    by_impl[name].append(_failure_row(suite, name, error))
                    continue
                by_impl[name] += [{**row, "impl": name} for row in child]

    rows = [row for name in impls for row in by_impl[name]]
    if settings.csv:
        write_csv(settings.csv, suite.schema, rows)
        log(f"wrote {settings.csv}")
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


def _pipeline_args(case: dict[str, Any], config: Path) -> list[str]:
    """What this case's codec pipeline adds to its implementation's environment.

    Per case rather than folded into `REQUIRES`, so an implementation measured
    on zarr-python's own pipeline resolves exactly the dependency tree it
    resolved before this axis existed. uv caches both, so the second environment
    costs an install once rather than once per case.
    """
    args: list[str] = []
    for token in _pipeline.REQUIRES.get(case.get("pipeline", ""), ()):
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
    runnable: dict[str, list[dict[str, Any]]],
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
        planned = runnable[name]
        # The pipeline's own pin included, because `--list` is read to find out
        # what is about to be installed and this axis adds to every column at
        # once. Listed once even though it is added per case.
        extra = sorted(
            {
                token
                for case in planned
                for token in _pipeline.REQUIRES.get(case.get("pipeline", ""), ())
            }
        )
        requires = " ".join((*getattr(adapter, "REQUIRES", ()), *extra))
        pin = getattr(adapter, "PYTHON", None)
        python = f" [python {pin}]" if pin else ""
        repeats = settings.repeats or getattr(adapter, "REPEATS", DEFAULT_REPEATS)
        runs = executions(repeats, settings.warmup)
        print(
            f"  {name:<{width}}  {len(planned):>3} cases x {runs} runs{python}  "
            f"{requires}"
        )
        for case in planned:
            variant = _variant(case)
            if variant:
                print(f"  {'':<{width}}    {case['op']} {case['image']}  {variant}")

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
    #
    # `total` is also the number of child processes, since each case gets its
    # own -- which is the other thing worth knowing before committing to a
    # sweep, and is not inferable from the case count of the old runner.
    print(
        f"\n{len([n for n, p in runnable.items() if p])} environments, "
        f"{total} cases in {total} child processes, "
        f"{settings.repeats or 'per-adapter'} repeats, "
        f"{settings.warmup} warmup"
    )
    effective = settings.repeats or DEFAULT_REPEATS
    if effective < 3:
        # Said here rather than discovered in the table: with fewer than three
        # timed runs there is no spread to report, and a number with no spread
        # beside it in a comparison reads as a precise one.
        print(
            f"note: repeats = {effective}; no run-to-run spread is estimable "
            "below 3, and the spread column will say so"
        )
    if settings.csv:
        print(f"csv: {settings.csv}")
    print()
