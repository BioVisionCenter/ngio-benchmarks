"""`ngio-bench-compare-create <config.toml> [--list]`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare._suite import Suite
from ngio_benchmarks.compare._suite import main as run_suite
from ngio_benchmarks.compare.create import IMPLS, OPS, SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec


def prepare(root: Path, specs: list[ImageSpec], ops: list[str]) -> None:
    """Write the shared pixel source every writer is handed.

    No read fixture here: each writer produces its own store, and comparing
    those stores is the experiment.
    """
    from ngio_benchmarks.core.data import write_source

    for spec in specs:
        write_source(root, spec)


SUITE = Suite(
    name="compare-create",
    prog="ngio-bench-compare-create",
    description="compare ngio's OME-Zarr writing against other OME-Zarr writers",
    impls=IMPLS,
    ops=OPS,
    schema=SCHEMA,
    default_images=("medium",),
    prepare=prepare,
)


def main(argv: list[str] | None = None) -> int:
    """Run the pyramid-creation comparison from a config file."""
    return run_suite(SUITE, argv)
