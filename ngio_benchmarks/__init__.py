"""Performance measurement for ngio, split in two halves that share a runner.

`internal` measures ngio against itself -- which option should I choose, what
does this cost at scale, did this version get slower. `compare` measures ngio
against the other libraries that read and write the same bytes.

The split is structural rather than cosmetic. A comparison peer's dependencies
do not reliably co-install with ngio's, so every comparison implementation runs
in its own isolated `uv` environment; the internal half runs in-process unless
a config asks otherwise. Mixing the two would drag that machinery into blocks
that have no use for it.

Nothing here imports ngio, or any peer library, at module scope. A child
process runs inside an environment holding exactly one implementation, and an
import at the wrong level would require every one of them to also hold ngio.
"""
