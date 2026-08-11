"""An experiment as a file. The only way anything here is configured.

A benchmark invocation is an experiment, and an experiment that lives only in
shell history cannot be committed next to the CSV it produced, diffed against
last month's, or pasted into an issue. So there are no tuning flags at all: a
suite takes a config path and `--list`, and everything else is in the file.

The rules the file obeys:

* **Every key is optional.** A file with nothing in it runs the defaults.
* **An unknown key is an error**, never ignored. A typo'd key that silently ran
  the defaults would be discovered only after the numbers had been believed.
* **A config replaces an axis; it never declares one.** Blocks stay the single
  source of truth for what is sweepable. `[images.<name>]` is not an exception
  -- see `core.images`.
* **Every axis entry names its block.** `[axes.consolidate] z = [16, 64]` and
  nothing shorter, because a file is read later by someone who was not there
  and "whichever blocks happen to have a `z`" describes the suite at the moment
  it was written rather than the experiment.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any, NamedTuple

from ngio_benchmarks.core import images as images_mod

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from ngio_benchmarks.core.axes import Override
    from ngio_benchmarks.core.images import ImageSpec

#: Keys every suite understands. A suite adds its own on top.
COMMON_KEYS = (
    "repeats",
    "warmup",
    "csv",
    "keep",
    "quiet",
    "axes",
    "images",
    "environments",
)

#: Used when a block declares no `REPEATS` of its own.
DEFAULT_REPEATS = 5


class EnvSpec(NamedTuple):
    """One environment to install and measure in.

    `ngio` names where ngio comes from; `requires` pins anything else. Being
    able to pin more than ngio is the point: installing an older ngio also
    resolves *its* dependency versions, so a difference between versions can
    come from zarr rather than from ngio, and pinning both is the only way to
    tell.
    """

    name: str
    ngio: str | None = None
    requires: tuple[str, ...] = ()
    python: str | None = None


class Common(NamedTuple):
    """The settings every suite shares, after validation."""

    repeats: int | None = None
    warmup: int = 1
    csv: Path | None = None
    keep: bool = False
    quiet: bool = False
    overrides: tuple[Override, ...] = ()
    images: dict[str, ImageSpec] = {}  # noqa: RUF012
    environments: tuple[EnvSpec, ...] = ()


def read(path: Path, keys: Collection[str]) -> dict[str, Any]:
    """Parse a TOML config and reject any key the suite does not know."""
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"no such config file: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(f"{path} is not valid TOML: {error}") from None

    unknown = sorted(set(data) - set(keys))
    if unknown:
        raise SystemExit(
            f"{path}: unknown key {unknown[0]!r}; "
            f"valid keys are {', '.join(sorted(keys))}"
        )
    return data


def typed(data: dict[str, Any], key: str, kind: type, where: Path) -> Any:
    """Fetch `key` from `data`, checking its type.

    `bool` is a subclass of `int`, so a plain `isinstance` check would let
    `repeats = true` through and then run some number of repeats nobody asked
    for. Checked explicitly rather than coerced -- a config is a record of an
    experiment, and quietly reinterpreting it defeats the point.
    """
    value = data.get(key)
    mistyped = not isinstance(value, kind) or (kind is int and isinstance(value, bool))
    if value is not None and mistyped:
        raise SystemExit(f"{where}: `{key}` must be {kind.__name__}, got {value!r}")
    return value


def names(data: dict[str, Any], key: str, valid: Collection[str], where: Path) -> Any:
    """Fetch a list-of-names key, checking every entry against `valid`."""
    value = typed(data, key, list, where)
    if value is None:
        return None
    if not value:
        raise SystemExit(f"{where}: `{key}` is empty; omit it to mean all of them")
    unknown = [v for v in value if v not in valid]
    if unknown:
        raise SystemExit(
            f"{where}: `{key}` has no {unknown[0]!r}; "
            f"choose from: {', '.join(sorted(valid))}"
        )
    return list(value)


def tokens(where: str, values: Any) -> list[str]:
    """Render one axis' values as tokens.

    Stringifying typed TOML into tokens looks like a loss, but it keeps a single
    code path: `apply_overrides` re-parses an open axis against the type its
    block declared, and a closed axis takes labels either way. Round-tripping an
    int, float, bool or string through `str` is exact.

    Booleans are the one exception, and are spelled the way TOML spells them.
    `str(True)` is `True`, so a config that wrote `use_tensorstore = [true]`
    would be told there is no `'True'` and offered `true` -- correct, unhelpful,
    and about a capital letter the file never contained.

    The scalar types are a whitelist, not a `list`/`dict` blacklist: TOML also
    has bare dates, and `z = 2026-08-10` would otherwise stringify happily, fail
    to parse as the int its axis declared, and sweep the leftover string without
    complaint.
    """
    if not isinstance(values, list) or not values:
        raise SystemExit(f"{where} must be a non-empty list of values")
    tokens = []
    for value in values:
        if not isinstance(value, bool | int | float | str):
            raise SystemExit(
                f"{where} takes numbers, strings or booleans, got "
                f"{type(value).__name__} ({value!r})"
            )
        tokens.append(str(value).lower() if isinstance(value, bool) else str(value))
    return tokens


def _overrides(axes: Any, blocks: Collection[str], where: Path) -> tuple[Override, ...]:
    """Lower an `[axes]` table into `(block, axis, tokens)` triples."""
    if not isinstance(axes, dict):
        raise SystemExit(f"{where}: `axes` must be a table, e.g. [axes.consolidate]")
    lowered: list[Override] = []
    for key, value in axes.items():
        if not isinstance(value, dict):
            raise SystemExit(
                f"{where}: axes.{key} is a bare axis, and every entry must name "
                f"its block. Write a [axes.<block>] table with `{key} = ...` "
                f"inside it; blocks are: {', '.join(sorted(blocks))}"
            )
        if key not in blocks:
            raise SystemExit(
                f"{where}: [axes.{key}] is not a block; "
                f"choose from: {', '.join(sorted(blocks))}"
            )
        for axis, values in value.items():
            lowered.append((key, axis, tokens(f"axes.{key}.{axis}", values)))
    return tuple(lowered)


def _images(table: Any, where: Path) -> dict[str, ImageSpec]:
    """Build the `[images.<name>]` additions to the registry."""
    if not isinstance(table, dict):
        raise SystemExit(f"{where}: `images` must be a table, e.g. [images.my_plate]")
    return {
        name: images_mod.from_toml(name, spec, where) for name, spec in table.items()
    }


def _environments(value: Any, where: Path) -> tuple[EnvSpec, ...]:
    """Build the `[[environments]]` list."""
    if not isinstance(value, list):
        raise SystemExit(f"{where}: `environments` must be a list of [[tables]]")
    specs = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise SystemExit(f"{where}: environments[{index}] must be a table")
        unknown = sorted(set(entry) - {"name", "ngio", "with", "python"})
        if unknown:
            raise SystemExit(
                f"{where}: environments[{index}] has unknown key {unknown[0]!r}; "
                "valid keys are name, ngio, with, python"
            )
        requires = tuple(entry.get("with", ()))
        ngio = entry.get("ngio")
        if ngio is None and not requires:
            raise SystemExit(
                f"{where}: environments[{index}] pins nothing; set `ngio` to a "
                'version ("1.0.0"), a checkout ("local:../ngio"), or a git ref '
                '("git+https://github.com/BioVisionCenter/ngio@main")'
            )
        specs.append(
            EnvSpec(
                name=entry.get("name") or ngio or ",".join(requires),
                ngio=ngio,
                requires=requires,
                python=entry.get("python"),
            )
        )
    return tuple(specs)


def common(data: dict[str, Any], path: Path, blocks: Collection[str]) -> Common:
    """Validate the keys every suite shares.

    A relative `csv` resolves against the config file's own directory rather
    than the working directory, so a committed experiment writes its results
    beside itself no matter where it is invoked from.
    """
    csv = typed(data, "csv", str, path)
    warmup = typed(data, "warmup", int, path)
    repeats = typed(data, "repeats", int, path)
    # Checked rather than tolerated: `repeats` is read downstream as
    # `repeats or <default>`, so a 0 would be falsy and silently run the
    # block's own repeat count instead of the none that was asked for.
    if repeats is not None and repeats < 1:
        raise SystemExit(f"{path}: `repeats` must be at least 1, got {repeats}")
    if warmup is not None and warmup < 0:
        raise SystemExit(f"{path}: `warmup` cannot be negative, got {warmup}")
    return Common(
        repeats=repeats,
        warmup=1 if warmup is None else warmup,
        csv=(path.parent / csv) if csv else None,
        keep=bool(typed(data, "keep", bool, path)),
        quiet=bool(typed(data, "quiet", bool, path)),
        overrides=_overrides(data.get("axes", {}), blocks, path),
        images=_images(data.get("images", {}), path),
        environments=_environments(data.get("environments", []), path),
    )
