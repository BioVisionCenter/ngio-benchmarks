"""What differs between the three suites' reports, as data.

One rendering engine, three descriptions of what it is rendering. The
alternative was three copies of `_model` and three copies of `report.js`, which
would have drifted the first time a formatting rule changed -- the same reason
`MEASUREMENT_FIELDS` is spliced into three schemas rather than retyped.

A profile holds three kinds of thing, and they are different kinds:

* **structure** -- which column names a series, which one groups, which axes
  claim a role. Read by `_model` to shape the payload.
* **audit** -- the column that decides whether two bars measured the same
  artefact. `compare-create` has `pyramid` and a literal value that means "what
  was asked for"; `compare-io` has `checksum` and no such value at all, because
  the answer is whatever the rest of the cell agreed on; `internal` has neither,
  because it measures one library against itself.
* **copy** -- what the page calls things. Shipped to the browser verbatim and
  never reassembled there, so every word a reader sees is in this file.

Nothing here imports an adapter or a block. It imports three `Schema`s, and a
`Schema` is a tuple of column names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from ngio_benchmarks.compare.create import SCHEMA as CREATE_SCHEMA
from ngio_benchmarks.compare.io import SCHEMA as IO_SCHEMA
from ngio_benchmarks.internal._run import SCHEMA as INTERNAL_SCHEMA

if TYPE_CHECKING:
    from ngio_benchmarks.core.output import Schema


class Detail(NamedTuple):
    """One line of the tooltip and of the pinned-row panel.

    `key` is a record key, empty for a pair the page computes rather than reads
    -- `spread` is a function of four columns. `format` names one of the
    formatters in `report.js`, all of which mirror `core.output`, so the page
    and the terminal cannot disagree about the same number.
    """

    label: str
    key: str
    #: spread | duration | megabytes | peak | bytes | times | text
    format: str


class Column(NamedTuple):
    """One trailing column of a table view, past the ones every suite has."""

    label: str
    key: str
    format: str
    numeric: bool = False


class Caveat(NamedTuple):
    """The audit column that decides whether two bars are comparable.

    `expected` is the value that means "comparable". An empty `expected` means
    there is no such value to name in advance and the comparison is against the
    rest of the cell instead: every implementation reading the same store should
    return the same digest, and which digest that is depends on the file.
    """

    field: str
    expected: str
    #: Drawn after the value label on a bar that fails the audit.
    mark: str
    #: Drawn under an `ok` cell in the coverage matrix.
    chip: str
    legend: str
    help: str
    #: `{n}` rows and `{s}` for the plural, filled in by `_model.shape`.
    notice: str


class Profile(NamedTuple):
    """One suite's report."""

    suite: str
    #: The console-script suffix: `ngio-bench-report-<command>`.
    command: str
    schema: Schema
    #: Which column records the library or ngio version, per row.
    version_field: str
    #: Axis candidates, in the order they are offered as roles and in the order
    #: they sort a series label. `schema.group` belongs in here -- `compare-io`
    #: has six operations and `internal` four blocks, and neither is a detail on
    #: a row. `compare-create` has exactly one operation, which is why the first
    #: version of this list could leave it out.
    candidates: tuple[str, ...]
    excluded: tuple[str, ...]
    facet_preference: tuple[str, ...]
    group_preference: tuple[str, ...]
    #: Series names pinned to palette slots, by index. A name not in here takes
    #: the lowest free slot, so an environment a config named itself still gets
    #: a colour of its own.
    palette: tuple[str, ...]
    #: The series a ratio is taken against by default; empty for a suite where
    #: no name leads.
    baseline: str
    #: `split` -- import cost plus case cost, for the suites that record both.
    #: `peak` -- tracemalloc's own figure, for the suite that records only it.
    memory: str
    details: tuple[Detail, ...]
    timing_columns: tuple[Column, ...]
    coverage_columns: tuple[Column, ...]
    #: A record key drawn under an `ok` cell in the coverage matrix, or empty.
    coverage_chip: str
    caveat: Caveat | None
    eyebrow: str
    title: str
    column_label: str
    group_label: str
    versions_label: str
    coverage_subhead: str
    memory_subhead: str
    legend: tuple[str, ...]
    #: Printed when a file holds more than one group value and the suite only
    #: defines one. Empty for the two suites where several are expected.
    mixed_group_notice: str


#: The paragraph under every timing legend. The two invariants it states are
#: the ones the whole page is built on, so it is one sentence in one place
#: rather than three that could drift apart.
_COLOUR_NOTE = (
    "Colour names the {column} and nothing else — it never encodes a value, "
    "and the slot is assigned by name, so hiding a series in the filter row "
    "never repaints the survivors. Colour is never the only channel either: "
    "every bar is directly labelled, every chart has a table view, and the "
    "texture toggle carries identity without hue at all."
)

_MEMORY_SPLIT = (
    "peak resident memory: what importing the library cost, and what the case "
    "cost on top"
)

#: The five pairs every measured row can show, in the order the tooltip reads
#: them: how noisy, how parallel, then the three memory figures.
_MEASURED = (
    Detail("spread", "", "spread"),
    Detail("cpu / wall", "parallelism", "times"),
    Detail("peak RAM", "peakMb", "peak"),
    Detail("peak RSS", "procPeakMb", "megabytes"),
    Detail("import cost", "rssBaseMb", "megabytes"),
)


CREATE = Profile(
    suite="compare-create",
    command="compare-create",
    schema=CREATE_SCHEMA,
    version_field="impl_version",
    candidates=("impl", "impl_version", "op", *CREATE_SCHEMA.axis_fields),
    # A description of what `method` became inside one library (`dask/linear`,
    # `decimate`), so it is detail on a row rather than an axis anyone sweeps.
    excluded=("method_native",),
    facet_preference=("image",),
    # `pipeline` after `method` rather than instead of it: a file that swept
    # only the codec pipeline then draws zarr-python and zarrs side by side
    # inside each library, which is the whole shape of that question.
    group_preference=("method", "pipeline"),
    palette=("ngio", "ngff-zarr", "ome-zarr-py", "bioio", "iohub", "acquire-zarr"),
    baseline="ngio",
    memory="split",
    details=(
        *_MEASURED,
        Detail("store", "bytes", "bytes"),
        Detail("levels", "levels", "text"),
        Detail("level shapes", "levelShapes", "text"),
        Detail("pyramid", "pyramid", "text"),
        Detail("codec", "codec", "text"),
        Detail("chunks", "chunks", "text"),
        Detail("shards", "shards", "text"),
        Detail("filter used", "methodNative", "text"),
        Detail("version", "version", "text"),
    ),
    timing_columns=(
        Column("store", "bytes", "bytes", True),
        Column("pyramid", "pyramid", "text"),
    ),
    coverage_columns=(
        Column("pyramid", "pyramid", "text"),
        Column("codec", "codec", "text"),
        Column("note", "note", "text"),
    ),
    coverage_chip="codec",
    caveat=Caveat(
        field="pyramid",
        expected="as asked",
        mark="≠",
        chip="≠ pyramid",
        legend="≠ built a different pyramid",
        help=(
            "A hatched bar marked ≠ wrote a pyramid other than the one it was "
            "asked for, so its timing is not like-for-like with the bars beside "
            "it. Check the pyramid, codec and store lines in the tooltip before "
            "reading its duration as a result."
        ),
        notice=(
            "{n} row{s} built a pyramid that differs from the one requested. "
            "Those bars are hatched — they timed a different artefact, so they "
            "are not like-for-like."
        ),
    ),
    eyebrow="ngio benchmarks · compare-create",
    title="Building an OME-Zarr, compared",
    column_label="implementation",
    group_label="operation",
    versions_label="versions",
    coverage_subhead="what each writer can express, and what it wrote when it could",
    memory_subhead=_MEMORY_SPLIT,
    legend=(_COLOUR_NOTE.format(column="implementation"),),
    mixed_group_notice=(
        "This file mixes operations ({values}). The suite only defines "
        "create_pyramid, so the charts pool them."
    ),
)


IO = Profile(
    suite="compare-io",
    command="compare-io",
    schema=IO_SCHEMA,
    version_field="impl_version",
    candidates=("impl", "impl_version", "op", *IO_SCHEMA.axis_fields),
    excluded=(),
    # `op` ahead of `image`, deliberately. A full-level read and a straddling
    # ROI write differ by orders of magnitude, and one card holding both under a
    # shared linear scale renders the fast operations as two-pixel slivers. One
    # card per operation gives each its own scale -- which is also how
    # `output.report_matrix` already nests the ASCII tables.
    facet_preference=("op", "image"),
    group_preference=("image",),
    palette=("ngio", "zarr", "zarrs", "dask", "tensorstore", "z5py", "acquire-zarr"),
    baseline="ngio",
    memory="split",
    details=(
        *_MEASURED,
        Detail("store", "bytes", "bytes"),
        Detail("checksum", "checksum", "text"),
        Detail("version", "version", "text"),
    ),
    timing_columns=(
        Column("store", "bytes", "bytes", True),
        Column("checksum", "checksum", "text"),
    ),
    coverage_columns=(
        Column("checksum", "checksum", "text"),
        Column("note", "note", "text"),
    ),
    coverage_chip="",
    caveat=Caveat(
        field="checksum",
        # No literal to compare against: the digest depends on the fixture, so
        # the only available claim is that everyone reading one store agreed.
        expected="",
        mark="≠",
        chip="≠ bytes",
        legend="≠ moved bytes the others did not",
        help=(
            "Every row in one cell handles the same pixels — a read opens the "
            "one shared store, a write is read back off disk afterwards — so all "
            "of them should hash to the same digest. A hatched bar marked ≠ did "
            "not: it read a different region, dtype or level, or it wrote "
            "something other than what it was given, and its duration is not "
            "comparable with the bars beside it. A blank cell is a write nothing "
            "could read back — unchecked, which is not the same as passing."
        ),
        notice=(
            "{n} row{s} produced a checksum the rest of the cell did not agree "
            "on. Those bars are hatched — they did not move the same bytes, so "
            "they are not like-for-like."
        ),
    ),
    eyebrow="ngio benchmarks · compare-io",
    title="Reading and writing an array, compared",
    column_label="implementation",
    group_label="operation",
    versions_label="versions",
    coverage_subhead="which operations each library expresses, and what they cost",
    memory_subhead=_MEMORY_SPLIT,
    legend=(_COLOUR_NOTE.format(column="implementation"),),
    mixed_group_notice="",
)


INTERNAL = Profile(
    suite="internal",
    command="internal",
    schema=INTERNAL_SCHEMA,
    version_field="ngio_version",
    # `block` leads and `env` trails, and the order is the point: series sort by
    # it, so with `block` spent on the facet the bars inside one card sort by
    # that block's own axes first and by environment last -- which puts the two
    # versions of the same case next to each other, and that adjacency is the
    # comparison `[[environments]]` exists to make.
    candidates=("block", *INTERNAL_SCHEMA.axis_fields, "env", "ngio_version"),
    excluded=(),
    facet_preference=("block",),
    # Nothing, on purpose. The four blocks sweep disjoint axes -- {mode, z},
    # {layout, z}, {alignment, size}, {kernel, n} -- so any single global group
    # axis is blank for three of them, and a card captioned `z = —` above a
    # block that has no `z` is worse than no caption at all. The picker still
    # offers every axis the file varies.
    group_preference=(),
    # Empty: an environment is whatever a config called it, so there is no name
    # to pin. Slots fall out in declaration order, which puts the first
    # environment -- `current` for a config with no `[[environments]]` -- on the
    # brand teal.
    palette=(),
    baseline="",
    memory="peak",
    details=(
        Detail("spread", "", "spread"),
        Detail("cpu / wall", "parallelism", "times"),
        Detail("peak RAM", "peakMb", "peak"),
        # Named for what it actually is. The internal blocks share one
        # interpreter, so this is the high-water mark across the whole run
        # wearing each row's label -- not this case's cost, which is the
        # tracemalloc figure above it.
        Detail("process peak RSS", "procPeakMb", "megabytes"),
        Detail("ngio", "version", "text"),
    ),
    timing_columns=(),
    coverage_columns=(Column("note", "note", "text"),),
    coverage_chip="",
    # No audit column. There is one implementation here, so there is no claim
    # that two bars built the same artefact to check.
    caveat=None,
    eyebrow="ngio benchmarks · internal",
    title="ngio measured against itself",
    column_label="environment",
    group_label="block",
    versions_label="ngio",
    coverage_subhead="which cases ran in each environment, and what they cost",
    memory_subhead="peak memory Python's allocator accounted for, per case",
    legend=(
        "Colour names the environment and nothing else. A file with one "
        "environment is one colour throughout, and what tells two bars apart "
        "there is the label: the case the block ran. Every chart also has a "
        "table view, and the texture toggle carries identity without hue.",
    ),
    mixed_group_notice="",
)


PROFILES = {profile.suite: profile for profile in (INTERNAL, IO, CREATE)}
