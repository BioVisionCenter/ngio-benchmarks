"""Isolated environments: run the same measurement against different installs.

One product; the outer factors are realized by re-exec, the inner ones
in-process. A library version cannot be varied inside a running interpreter, so
the parent spawns one child per environment and each child enumerates its cases
normally.

The comparison half uses this unconditionally, one environment per
implementation. That is not caution about a conflict that might appear: `z5py`
is zarr v2 only, `iohub` needs Python 3.12 while ngio supports 3.11, and
`ngff-zarr` and `ome-zarr-py` pin dask ranges that intersect in a single narrow
window. There is no single environment in which all of them are installable,
and a suite that could only measure the subset that happens to co-install today
would quietly shrink every time one of them released.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ngio_benchmarks.core.cli import PROJECT_ROOT
from ngio_benchmarks.core.output import log, read_csv

if TYPE_CHECKING:
    from ngio_benchmarks.core.config import EnvSpec
    from ngio_benchmarks.core.job import Job


def require_uv() -> None:
    """Fail early, with the install link, if `uv` is missing."""
    if shutil.which("uv") is None:
        raise SystemExit(
            "this needs `uv` on PATH (https://docs.astral.sh/uv/).\n"
            "Every measured library is installed into its own isolated "
            "environment, which is what uv is doing here."
        )


def requirement(token: str, base: Path, *, package: str | None = None) -> list[str]:
    """Translate one requirement token into `uv run` arguments.

    Three sources, because those are the three places a version anyone wants to
    measure actually lives:

    | token                          | means                                |
    | ------------------------------ | ------------------------------------ |
    | `1.0.0` or `>=1.0`             | that release from PyPI               |
    | `git+https://.../ngio@main`    | that git ref                         |
    | `local:../ngio`                | that checkout, installed editable    |

    A relative `local:` path resolves against `base` -- the config file's own
    directory -- so a committed experiment points at a sibling checkout the same
    way from any working directory.
    """
    if token.startswith("local:"):
        path = (base / token[len("local:") :]).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"no such checkout: {path} (from {token!r})")
        return ["--with-editable", str(path)]
    if token.startswith(("git+", "http://", "https://")):
        if package is None:
            raise SystemExit(
                f"{token!r} is a URL, which only the `ngio` key accepts; under "
                "`with` write a PEP 508 requirement such as "
                "'ngio @ git+https://github.com/BioVisionCenter/ngio@main'"
            )
        return ["--with", f"{package} @ {token}"]
    if package is not None and token[:1].isdigit():
        # A bare `1.0.0` under the `ngio` key is the common case and reads far
        # better in a committed file than `ngio==1.0.0`.
        return ["--with", f"{package}=={token}"]
    return ["--with", token]


def uv_args(env: EnvSpec, base: Path) -> list[str]:
    """Every `--with` argument one environment needs."""
    args: list[str] = []
    if env.ngio:
        args += requirement(env.ngio, base, package="ngio")
    for token in env.requires:
        args += requirement(token, base)
    return args


def run_child(
    label: str,
    job: Job,
    with_args: list[str],
    *,
    python: str | None = None,
    quiet: bool = False,
    index: int | None = None,
    total: int | None = None,
) -> tuple[list[dict[str, str]], str]:
    """Install one environment, run one job in it, and read back its rows.

    Returns the rows and an error string, empty when it worked. A failure is
    reported and skipped rather than aborting: with six implementations in a
    comparison, one that will not install today must not delete the other five.

    The child's result file is placed here, in this call's own temp directory,
    and whatever `out` the caller set is overridden. It is an intermediate, not
    data: a child appends to it, so under `keep` a path inside the persisted
    data root would collect every previous run's rows and the parent would read
    them all back as if they were this run's.

    `--no-project` is load-bearing. `uv run` inside a directory containing a
    `pyproject.toml` installs *that* project by default, and the working
    directory here is necessarily the project root -- that is how the child
    imports `ngio_benchmarks` without it being installed. Without the flag,
    every environment would also get this package's own dependencies.
    """
    interpreter = python or f"{sys.version_info.major}.{sys.version_info.minor}"
    prefix = f"[{index}/{total}] " if index is not None and total is not None else ""
    with tempfile.TemporaryDirectory(prefix="ngio-bench-job-") as tmp:
        out = Path(tmp) / "out.csv"
        path = job._replace(out=str(out)).write(Path(tmp) / "job.json")
        command = [
            "uv", "run", "--no-project", "--python", interpreter,
            *with_args,
            "python", "-m", "ngio_benchmarks._child", str(path),
        ]  # fmt: skip
        if not quiet:
            log(f"{prefix}=> {label}")
        completed = subprocess.run(
            command, check=False, cwd=PROJECT_ROOT, capture_output=quiet
        )
        if completed.returncode != 0 or not out.exists():
            detail = f"exit {completed.returncode}"
            if quiet and completed.stderr:
                detail += (
                    f": {completed.stderr.decode(errors='replace').strip()[-400:]}"
                )
            log(f"{prefix}   {label}: FAILED ({detail})")
            return [], detail
        return read_csv(out), ""


def describe(command: list[str]) -> str:
    """Render a uv command for `--list`, so a dry run shows what will install."""
    return " ".join(json.dumps(part) if " " in part else part for part in command)
