"""The internal suite: ngio measured against itself.

Answers *which option should I choose*, *what does this cost at scale*, and
with `[[environments]]`, *did this version get slower*. The comparison suites
answer a different question and share only the runner.

`BLOCKS` is a plain dict of module *paths*, not imported modules. Inside an
environment holding an older ngio a block may fail to import, and that has to
degrade one block rather than the whole run -- so the runner imports each
selected block lazily. One consumer, so a registry would be a dict with extra
steps.

A block earns its place by informing a decision. Blocks that merely produce a
number do not. Keep this list short.
"""

BLOCKS = {
    "consolidate": "ngio_benchmarks.internal.blocks.consolidate",
    "layout": "ngio_benchmarks.internal.blocks.layout",
    "roi": "ngio_benchmarks.internal.blocks.roi",
    "algorithms": "ngio_benchmarks.internal.blocks.algorithms",
    "iterators": "ngio_benchmarks.internal.blocks.iterators",
}
