"""The entry-point scaffolding every suite shares.

There are exactly two things on a command line: the config file, and `--list`.
Everything that shapes an experiment lives in the file, so there is no
precedence to define, no `--no-` forms so a `true` in a file can be switched
off, and no lowering of flags into a child process.

`--list` survives because it is not a setting -- it is the dry run. With
several isolated environments to install before the first measurement, finding
out after the install phase that a config selected nothing is the expensive
failure mode.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The project root: the directory holding `pyproject.toml` and the
#: `ngio_benchmarks` package. A child process runs with this as its working
#: directory so `python -m ngio_benchmarks._child` imports the package from
#: `sys.path[0]` without it being installed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse(prog: str, description: str, argv: list[str] | None) -> argparse.Namespace:
    """Parse the two arguments a suite accepts."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("config", type=Path, help="the experiment, as a TOML file")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print what this config would run, then exit without measuring",
    )
    return parser.parse_args(argv)


@contextmanager
def data_root(keep: bool) -> Iterator[Path]:
    """Yield the directory generated fixtures live in.

    Under `keep` it is `.data/` beside the project, reused across runs --
    fixture names derive from their spec, so two blocks asking for the same
    image share one store. Otherwise it is a temp directory removed on the way
    out.
    """
    if keep:
        root = PROJECT_ROOT / ".data"
        root.mkdir(parents=True, exist_ok=True)
        yield root
        return
    tmp = tempfile.mkdtemp(prefix="ngio-bench-")
    try:
        yield Path(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
