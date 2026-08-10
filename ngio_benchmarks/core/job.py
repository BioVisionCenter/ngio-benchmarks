"""What a parent hands a child process.

The user-facing config is TOML; this is not. A job is machine-generated, read
once, and never edited, so it is JSON: writing TOML needs a dependency the
child environments must not be made to carry, and a format nobody hand-edits
gains nothing from TOML's ergonomics.

Passing a resolved job rather than re-serialising the config has two
consequences worth naming. A child is told exactly one environment or one
implementation, which is what stops the parent form recursing. And an axis
value never round-trips through a comma-joined command line, so -- unlike the
flag-lowering this replaces -- a value containing a comma is simply a value.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple

from ngio_benchmarks.core import images as images_mod

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.axes import Override
    from ngio_benchmarks.core.images import ImageSpec


class Job(NamedTuple):
    """One child's entire instruction set."""

    suite: str
    #: The environment or implementation name this child stands for. The parent
    #: knows it; the child does not need to, but it is recorded so the CSV a
    #: child writes is already labelled and the merge is a concatenation.
    label: str
    root: str
    #: Where the child writes its rows. Filled in by `run_child`, which puts it
    #: in that call's own temp directory -- a caller cannot place it, because
    #: the one plausible place to put it (the data root) is wrong under `keep`.
    out: str = ""
    repeats: int | None = None
    warmup: int = 1
    blocks: tuple[str, ...] = ()
    ops: tuple[str, ...] = ()
    impl: str = ""
    overrides: tuple[Override, ...] = ()
    images: dict[str, ImageSpec] = {}  # noqa: RUF012

    def write(self, path: Path) -> Path:
        """Serialise to `path` and return it."""
        data: dict[str, Any] = {
            "suite": self.suite,
            "label": self.label,
            "root": self.root,
            "out": self.out,
            "repeats": self.repeats,
            "warmup": self.warmup,
            "blocks": list(self.blocks),
            "ops": list(self.ops),
            "impl": self.impl,
            "overrides": [[b, a, list(t)] for b, a, t in self.overrides],
            "images": {name: spec.to_dict() for name, spec in self.images.items()},
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def read(cls, path: Path) -> Job:
        """Load a job written by `write`."""
        data = json.loads(path.read_text())
        return cls(
            suite=data["suite"],
            label=data["label"],
            root=data["root"],
            out=data["out"],
            repeats=data["repeats"],
            warmup=data["warmup"],
            blocks=tuple(data["blocks"]),
            ops=tuple(data["ops"]),
            impl=data["impl"],
            overrides=tuple((b, a, list(t)) for b, a, t in data["overrides"]),
            images={
                name: images_mod.from_toml(name, spec, path)
                for name, spec in data["images"].items()
            },
        )
