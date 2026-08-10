"""Parameterisation as data: axis declarations, their product, and overrides.

A block declares its axes as data rather than burying literals in a loop, and
the runner measures every point of their cartesian product. That is the whole
design, and it is what lets a config file sweep an axis and have the values
land as real CSV columns.

An axis is declared in one of two forms, and the form is the meaning:

| declaration                      | kind   | a config can                   |
| -------------------------------- | ------ | ------------------------------ |
| `"z": [16, 64, 256]`             | open   | subset, and add new values     |
| `"layout": {"sharded": {...}}`   | closed | subset only                    |

Values on an open axis are scalars, so `z = [1024]` re-parses as an int and
sweeps a depth nobody declared. Values on a closed axis are structures -- a
bundle of `create_empty_ome_zarr` kwargs, say -- so its labels are the
vocabulary and a shard shape is not something to spell out per experiment.

The `image` axis is closed and draws its values from a registry that a config
file may extend (see `core.images`). That is not a hole in the rule above: the
block still declares the axis, and the config adds vocabulary to the registry
the axis draws from.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

#: One axis override: `(block, axis name, value tokens)`.
#:
#: The block is never `None`. The old command line accepted an unqualified
#: `--axis z=16` that reached every block declaring a `z`; a config file cannot,
#: because a file is read later by someone who was not there and "whichever
#: blocks happen to have a `z`" describes the suite at the moment it was
#: written rather than the experiment.
Override = tuple[str, str, list[str]]


class Case(NamedTuple):
    """One point of a block's cartesian product."""

    block: str
    labels: dict[str, str]
    values: dict[str, Any]

    @property
    def label(self) -> str:
        """`mode=numpy z=256`, or the block name when there are no axes."""
        return " ".join(f"{k}={v}" for k, v in self.labels.items()) or self.block


def normalize(axis: list[Any] | Mapping[str, Any]) -> dict[str, Any]:
    """Return `{label: value}` for either declaration form."""
    return dict(axis) if isinstance(axis, dict) else {str(v): v for v in axis}


def cases(block: str, axes: Mapping[str, Any]) -> Iterator[Case]:
    """Enumerate the cartesian product of `axes`; the last varies fastest."""
    normalized = {name: normalize(axis) for name, axis in axes.items()}
    names = list(normalized)
    for point in product(*(normalized[name].items() for name in names)):
        yield Case(
            block,
            {n: label for n, (label, _) in zip(names, point, strict=True)},
            {n: value for n, (_, value) in zip(names, point, strict=True)},
        )


def _coerce(sample: Any, raw: str) -> Any:
    """Parse `raw` as the type of `sample`, falling back to the string."""
    if isinstance(sample, bool):
        return raw.lower() in ("1", "true", "yes")
    for kind in (type(sample), int, float):
        try:
            return kind(raw)
        except (TypeError, ValueError):
            continue
    return raw


def apply_overrides(
    block: str,
    axes: Mapping[str, Any],
    overrides: tuple[Override, ...],
) -> dict[str, dict[str, Any]]:
    """Return `{axis: {label: value}}` with this block's overrides applied.

    An override *replaces* an axis outright rather than extending it. On a
    closed axis it names labels; on an open axis it names values, re-parsed
    with the type of the first declared one so a value that was never declared
    can still be swept.

    An override naming an axis the block does not declare is an error, not a
    silent no-op: a config that quietly measured the defaults would be
    discovered only after its numbers had been believed.
    """
    resolved = {name: normalize(axis) for name, axis in axes.items()}
    for target, name, raw in overrides:
        if target != block:
            continue
        if name not in resolved:
            raise SystemExit(
                f"{block} has no axis {name!r}; it has: {', '.join(resolved) or 'none'}"
            )
        declared = resolved[name]
        if isinstance(axes[name], dict):
            unknown = [token for token in raw if token not in declared]
            if unknown:
                raise SystemExit(
                    f"{block}.{name} has no {unknown[0]!r}; "
                    f"choose from: {', '.join(declared)}"
                )
            resolved[name] = {token: declared[token] for token in raw}
        else:
            sample = next(iter(declared.values()), "")
            resolved[name] = {token: _coerce(sample, token) for token in raw}
    return resolved


def count(axes: Mapping[str, Mapping[str, Any]]) -> int:
    """How many cases a resolved axis mapping enumerates."""
    total = 1
    for values in axes.values():
        total *= len(values)
    return total
