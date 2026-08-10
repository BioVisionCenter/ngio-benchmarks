"""`ngio-bench-compare-io <config.toml> [--list]`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare._suite import Suite
from ngio_benchmarks.compare._suite import main as run_suite
from ngio_benchmarks.compare.io import IMPLS, OPS, SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec


def prepare(root: Path, specs: list[ImageSpec], ops: list[str]) -> None:
    """Build everything the children share, before any of them installs.

    The read fixture is written here rather than by the first child that wants
    it, so that no implementation's environment has to carry zarr-python purely
    for setup, and so a fixture is never half-written by a child that failed.
    """
    from ngio_benchmarks.compare.io._fixture import build
    from ngio_benchmarks.core.data import write_source

    for spec in specs:
        if any(op.startswith("write") for op in ops):
            write_source(root, spec)
        if any(op.startswith("read") for op in ops):
            build(root, spec)


SUITE = Suite(
    name="compare-io",
    prog="ngio-bench-compare-io",
    description="compare ngio's array access against other chunked-array libraries",
    impls=IMPLS,
    ops=OPS,
    schema=SCHEMA,
    default_images=("medium",),
    prepare=prepare,
)


def main(argv: list[str] | None = None) -> int:
    """Run the io comparison from a config file."""
    return run_suite(SUITE, argv)
