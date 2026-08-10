# ngio-benchmarks

Performance measurement for [ngio](https://github.com/BioVisionCenter/ngio),
split into two halves that answer different questions and share a runner.

**Internal** measures ngio against itself — *which option should I choose*,
*what does this cost at scale*, *did this version get slower*.

**Comparison** measures ngio against the other libraries that read and write the
same bytes — *what does ngio's layer cost over raw zarr*, and *how does building
an OME-Zarr with ngio compare to building one with anything else*.

The split is structural rather than cosmetic. A comparison peer's dependencies
do not reliably co-install with ngio's, so every comparison implementation runs
in its own isolated `uv` environment. Mixing that machinery into blocks with no
use for it is what the separation avoids.

```bash
uv sync --extra internal

uv run ngio-bench-internal       experiments/smoke-internal.toml
uv run ngio-bench-compare-io     experiments/smoke-io.toml
uv run ngio-bench-compare-create experiments/smoke-create.toml
```

Nothing here is committed and nothing gates. The numbers depend on the machine;
compare within one run rather than across weeks.

## The config file is the only interface

There are no tuning flags. A suite takes a config path and `--list`, and
everything that shapes an experiment lives in the file:

```bash
uv run ngio-bench-compare-io experiments/my-sweep.toml --list   # dry run
uv run ngio-bench-compare-io experiments/my-sweep.toml
```

An invocation is an experiment, and an experiment that lives only in shell
history cannot be committed next to the CSV it produced, diffed against last
month's, or pasted into an issue. So there is no precedence to define, no
`--no-` forms so a `true` in a file can be switched off, and nothing an
invocation can add that the file does not record.

`--list` survives because it is not a setting — it is the dry run. It prints
the images, the implementations, the environments that will be installed, the
cases that will run and the ones that cannot, and then exits. **Run it before
any real sweep.** With one environment per implementation to install before the
first measurement, a config that selects nothing is expensive to discover any
later than this.

Three rules the files obey:

- **Every key is optional.** A file with nothing in it runs the defaults.
- **An unknown key is an error**, never ignored. A typo'd key that silently ran
  the defaults would be discovered only after the numbers had been believed.
- **A config replaces an axis; it never declares one.** The blocks and adapters
  stay the single source of truth for what is sweepable.

`experiments/reference-*.toml` document every key with its default, entirely
commented out — copy one and uncomment what you need. Because every key is
optional, a reference file as shipped is exactly equivalent to the defaults,
which makes `--list` on it a way to see what those defaults are.

## Parameterisation as data

A block declares its axes as data rather than burying literals in a loop, and
the runner measures every point of their cartesian product. That is what lets a
config file sweep an axis and have the values land as real CSV columns.

Axes come in two kinds, and the declaration form is the meaning:

| form | kind | a config can |
| --- | --- | --- |
| `"z": [16, 64, 256]` | **open** | subset, *and* name values never declared |
| `"layout": {"sharded": {...}}` | **closed** | subset only |

Open values are scalars, so `z = [1024]` re-parses as an int and sweeps a depth
nobody declared. Closed values are structures — a bundle of kwargs — so their
labels are the vocabulary. That is deliberate: chunk and shard shapes are not
independent knobs (`sharded` + `uncompressed` is not a case anyone wants), and a
shard shape is not something to respell in every experiment.

Two conventions run through the whole config surface. A **singular** key is an
axis and holds a list of labels; a **plural** table defines what those labels
mean — `image` selects, `[images.<name>]` defines; `impl` selects,
`[env.<impl>]` pins. And in the internal suite every axis entry names its block
(`[axes.consolidate] z = [16, 64]`), because a bare `z` meaning "whichever
blocks happen to have one" describes the suite at the moment the file was
written rather than the experiment.

## Images are a named registry

Everything that touches real data sweeps a closed `image` axis whose labels name
entries in a registry. Shipped: `small`, `medium`, `large`, `chunks64`,
`chunks512`, `sharded`, `uncompressed`, `v2`.

A config adds its own with an `[images.<name>]` table:

```toml
image = ["medium", "my_plate"]

[images.my_plate]
shape       = [1, 64, 2048, 2048]
chunks      = [1, 1, 256, 256]
shards      = [1, 8, 1024, 1024]
dtype       = "uint16"
compressors = "zstd"        # auto / none / zstd / blosc / lz4
zarr_format = 3             # 2 (NGFF 0.4) or 3 (NGFF 0.5)
levels      = 5
roi_size    = 512
```

This is not a hole in the "a config never declares an axis" rule: the block
still declares `image`, and it is still closed. The table adds a value to the
registry the axis draws from. The distinction matters because a spec is a bundle
of interdependent settings — shards imply zarr v3, a shard shape must be a whole
number of chunks — and a bundle wants validating once, by name, rather than
re-derived at every use site. All of that is checked when the file is read, so a
bad spec fails before anything installs rather than after.

The pixels are seeded, spatially correlated, and deliberately *compressible*.
Uniform data compresses ~2000:1 and pure noise 1:1, either of which would make a
comparison of chunk shapes, codecs or writers meaningless without looking wrong.
The generator is tuned to ~1.8x, matching the ~1.7x of a real sample image.

## Suite 1 — `internal`

The four blocks, and the decision each one informs:

| block | axes | question |
| --- | --- | --- |
| `consolidate` | `mode` × `z` | which pyramid mode should I use, and will it fit? |
| `layout` | `layout` × `z` | same bytes, different chunk/shard shape |
| `roi` | `alignment` × `size` | chunk-aligned vs straddling reads |
| `algorithms` | `kernel` × `n` | scaling curves for ngio's own algorithms |

`algorithms` reports a series so the *shape* is visible — one timing cannot tell
O(n) from O(n²), four can. Its three kernels have different useful `n` ranges, so
cases outside a kernel's range are skipped and the run reports how many it
dropped; a bounded sweep must never read as a complete one.

`get`/`set` are deliberately absent: they are a thin layer over zarr, so timing
them mostly re-measures zarr. What that layer costs is `compare-io`'s question.

### Peak memory is the point, not an extra

The three consolidation modes differ mainly in what they hold at once, so a
timing alone cannot tell you which one survives your data. `numpy` peaks at
roughly 1.5x the data and scales with it — a hard ceiling. `dask` and `coarsen`
are chunk-bounded and stay flat. `numpy` is also the *fastest*, which is only a
sensible trade if the level fits in RAM.

Timing and allocation are measured in **two separate phases**. `tracemalloc`
hooks every allocation and inflates allocation-heavy code, and it does not
inflate every case equally — the mode that allocates most is penalised most,
which is exactly the comparison this block exists to make. Peak is the max over
its runs, not the last one, because the question peak answers is "will this fit",
and that is a worst case.

That second phase is why **`warmup + repeats` is not how many times a case
runs**:

```
runs/case = warmup + repeats + min(2, repeats)
```

`repeats = 3, warmup = 1` is 6 runs, not 4. The allocation passes are capped at
`repeats`, so `repeats = 1` costs one run of each phase rather than three of
something you did not ask for. `--list` prints `runs/case` for every block and
implementation — worth checking before a sweep where one run builds a pyramid.

### What each memory column actually measures

Two columns, because no single method covers both halves of this project.

**`peak_mb` — `tracemalloc`.** Accurate for anything allocating through Python
or numpy, which is all of the `internal` suite. numpy registers a `tracemalloc`
domain and reports its data buffers through it, so this is an absolute figure
rather than merely a relative signal: on a 128 MiB read it says 132.5 MB where
the OS says 133.3 MB.

It is **blind to allocations a C++ or Rust extension makes on its own**. The
same 128 MiB read through tensorstore reports 0.0 MB. Implementations in that
position (`tensorstore`, `z5py`, `zarrs`, `acquire-zarr`) declare `NATIVE = True`
and their cell prints `n/a` — a `0.0 MB` there would read as "uses no memory",
which is a claim, and a wrong one.

**`proc_peak_mb` — peak RSS of the child process.** The OS high-water mark, so
it counts every byte including native buffers. Reported once per process, under
the tables, rather than per case — because RSS only rises. A peak-RSS delta
around a single call is exact the first time and reads ~0 afterwards, since the
allocator does not return the pages: measured five times on the same 128 MiB
read, the first call shows 133.5 MB and every later one shows 0.1 MB. It is
absolute, including the interpreter and the library's imports, which is part of
what running that implementation costs.

Exact per-case native peaks would need a fresh process per case. That is not
done: the comparison suites already spawn one process per implementation, and
one per case would multiply that by the case count for a number only four
columns need.

### Comparing versions

Omit `[[environments]]` and the blocks run in the current interpreter. Declare
some and each is installed into its own isolated environment and measured
separately, printed as one column per entry with a ratio:

```toml
blocks = ["roi"]

[[environments]]
name = "worktree"
ngio = "local:../../ngio"          # a checkout, installed editable

[[environments]]
name = "released"
ngio = "1.0.0"                     # a PyPI release
with = ["zarr==3.1.6"]             # pin its dependencies too

[[environments]]
name = "main"
ngio = "git+https://github.com/BioVisionCenter/ngio@main"
```

Being able to pin *anything*, not just ngio, is the point: installing an older
ngio also resolves *its* dependency versions, so a difference between versions
can come from zarr rather than from ngio. Pinning both is the only way to tell.
The CSV records the resolved `ngio_version` and `zarr` on every row for the same
reason — check those columns before concluding anything about an ngio change.

`keep` is not honoured across environments: a fixture written by ngio 1.0.0 and
read back by the working tree is a different experiment from the one that was
asked for, so every environment gets its own temp root.

## Suite 2 — `compare-io`

Array access compared across libraries, over `impl` × `op` × `image`.

| operation | what it does |
| --- | --- |
| `read_full` | read level 0 back as an array |
| `read_roi_aligned` | read a region whose bounds fall on chunk edges |
| `read_roi_straddling` | read the same-sized region offset into the chunks |
| `write_full` | write the whole array into a fresh store |
| `write_roi_aligned` | write one aligned region |
| `write_roi_straddling` | write one straddling region |

| impl | what it is |
| --- | --- |
| `ngio` | `open_image` / `get_as_numpy` / `set_array`, via public slicing keywords |
| `zarr` | zarr-python directly — the floor every other row is read against |
| `zarrs` | the same code under the zarrs (Rust) codec pipeline; **zarr v3 only** |
| `dask` | `da.from_array` / `da.store`, with `.compute()` inside the timing |
| `tensorstore` | the same bytes decoded outside Python — a ceiling, not a peer |
| `z5py` | a C++ zarr/n5 implementation with an h5py-shaped API |
| `acquire-zarr` | **write-only, streaming**; `write_full` and nothing else |

The aligned/straddling pair is the read-amplification question asked of
everyone: a region inside chunk boundaries touches the minimum number of chunks,
and the same region offset by half a chunk fetches and discards every chunk on
each edge. The ratio between the two columns is what that costs.

### One fixture, written by nobody in the running

Every read operation opens the **same store**, built by the parent with plain
zarr-python plus hand-written NGFF metadata. If the fixture were written by ngio,
ngio would be the only contestant reading a file produced by its own encoder, and
any difference in layout or codec would be a difference this suite created rather
than measured. Peers open the level-0 array directly and ignore the metadata;
ngio opens the container properly.

That the bytes really are the same is a column, not an assumption: every read
records a `checksum` of what it returned, computed once outside the timing.

## Suite 3 — `compare-create`

One operation — build a multiscale OME-Zarr from an array in memory — over
`impl` × `image`.

| impl | what it is |
| --- | --- |
| `ngio` | `create_empty_ome_zarr` + `set_array` + `consolidate` |
| `ngff-zarr` | `to_ngff_image` / `to_multiscales` / `to_ngff_zarr` (itkwasm filter) |
| `ome-zarr-py` | `write_image` — the OME project's own writer |
| `bioio` | `OMEZarrWriter`, told explicit per-level shapes |
| `iohub` | an HCS-shaped writer on a single position; **needs Python ≥3.12** |
| `acquire-zarr` | streams frames, building the pyramid during acquisition; **v3 only** |

### The pyramid has to be pinned, or nothing is comparable

Left to their own defaults these writers build **three different pyramids**:

| geometry | writers | level 1 of a (1,16,512,512) image |
| --- | --- | --- |
| xy only | `ngio`, `ome-zarr-py`, `bioio` | (1,16,**256,256**) |
| xyz | `ngff-zarr`, `iohub` | (1,**8**,256,256) |
| its own | `acquire-zarr` | (1,8,256,256), then xy stops halving |

The xyz writers do half the work at every level, so timing them against the xy
writers compares pyramid shapes rather than implementations. `spec.downsample`
settles it:

```toml
[images.my_plate]
shape      = [1, 64, 2048, 2048]
levels     = 5
downsample = ["y", "x"]        # default; or ["z", "y", "x"] for isotropic
```

Five of the six writers are pinned to it — ngio via `scaling_factors`,
ome-zarr-py and ngff-zarr via per-axis factor dicts, bioio via explicit level
shapes, iohub via `dims`. `acquire-zarr` cannot be: its streaming writer exposes
the filter and the depth but not the axes.

Only the axes are configurable; the factor is fixed at 2 per level, because
several of these writers can express nothing else and a setting only some
columns could honour is the problem, not the fix.

### The audit columns

`levels`, `level_shapes` and `pyramid` are read back **off disk**, not taken
from what the writer was asked for — asking it would not catch a writer that did
something else. `pyramid` says `as asked` or spells out the divergence, so the
one writer that cannot comply is labelled rather than silently different.

This has already earned its place twice: it caught `acquire-zarr` reading
`max_levels` as the count of levels *below* level 0, and it is how the
xy-versus-xyz split above was found at all — before pinning, it showed only as
`ngff-zarr` producing a 3.9 MB store where ngio produced 7.2 MB, which is easy
to misread as a codec difference. Pinned, both write ~7.18 MB and the remaining
spread (`bioio` 4.2 MB, `iohub` 5.7 MB) is genuinely chunking and codecs.

When the arrays on disk and the NGFF metadata disagree, both numbers are
reported: a store with five arrays declaring three is a different fault from
writing three.

Filter choice is still not pinned — the libraries do not offer a common one —
so `downsample` records what each actually used. Read it alongside the timing.

## Why every implementation gets its own environment

Unconditionally, not as a fallback. There is no single environment all of these
install into:

- `iohub` requires Python ≥3.12, while ngio supports 3.11.
- `ome-zarr` requires Python >3.11.
- `zarrs` is a zarr v3 codec pipeline with no v2 equivalent, and `acquire-zarr`
  0.8 dropped zarr v2 entirely.
- `z5py` publishes no manylinux aarch64 wheel.
- `ngff-zarr` and `ome-zarr-py` pin dask ranges that intersect only narrowly.

A suite that measured whichever subset happened to co-install today would quietly
shrink every time one of them released. This project tried shipping a `compare`
extra listing all of them; it does not resolve, and that failure is the premise
of the design rather than an obstacle to it.

Each adapter declares its own pins, and a config can replace them — which is how
a committed experiment names the exact peer version its numbers came from rather
than "whatever resolved the day it ran":

```toml
[env.ome-zarr-py]
requires = ["ome-zarr==0.18.0"]

[env.ngio]
requires = ["local:../../ngio"]     # measure the working tree
```

Requires `uv` on PATH. The first run pays a real install for every environment;
later runs hit uv's cache. An environment that fails to install is reported and
skipped rather than aborting the rest.

### A blank cell is a claim

In a comparison table an unexplained gap reads as a result, so the reasons a cell
has no number stay distinguishable:

| cell | means |
| --- | --- |
| `unsupported` | the adapter declares it cannot express this case |
| `unavailable` | the environment failed to install, or the import failed |
| `failed` | it ran and raised; the CSV `note` has the exception |
| `—` | not selected by this config |

`SUPPORTS` and `FORMATS` on each adapter are declarations, not documentation:
`--list` prints every excluded case with its reason before anything installs.

## Output

One CSV schema per suite, not a union across all three — the suites are separated
precisely so they are not read together. Within a suite the environment is
repeated on every row, so a single file answers cross-environment questions with
no joins, and runs append so several accumulate. A file whose header does not
match **raises** rather than being appended to.

Axis columns hold **labels**, not values: `layout` is the string `sharded`,
`image` is `medium`, never a repr of a kwargs dict. A CSV cell and a config token
are the same token, so `df.pivot(index="image", columns="impl", values="seconds")`
is a one-liner.

Results are gitignored (`experiments/**/*.csv`) while the toml is not. The recipe
is the committable artefact; the numbers are machine-dependent.

## Adding to it

A block, an operation or an implementation earns its place by informing a
decision — *is this doing more work than it should*, *which option should I
choose*, *is ngio competitive here*. Ones that merely produce a number do not.
Keep the lists short.

**A new internal block** is one file in `ngio_benchmarks/internal/blocks/`, one
line in `internal/__init__.py`, and any new axis name added to `AXIS_FIELDS` in
`internal/_run.py`. It declares `AXES`, an optional `REPEATS`, and:

```python
def run(root: Path, **values) -> Measured:
    """Set up, then return the callable to measure."""
```

Everything outside the returned callable is excluded from the measurement. It is
one function rather than a `(setup, run)` pair because setup and the measured
call always share state.

**A new comparison implementation** is one module in the relevant `adapters/`
directory and one line in that suite's `IMPLS`:

```python
NAME     = "tensorstore"
REQUIRES = ("tensorstore>=0.1.85", "numpy>=2")   # its uv environment
SUPPORTS = frozenset({"read_full", ...})         # operations it can express
FORMATS  = frozenset({2, 3})                     # zarr formats it handles
PYTHON   = None                                  # interpreter pin, if any

def build(op, spec, root) -> Measured: ...
```

The constants are read by the **parent**, which has none of these libraries
installed — so an adapter must import with nothing but the standard library and
numpy. Every `import tensorstore` belongs inside `build`, never at module scope.
The same discipline applies to `ngio_benchmarks/core/`, which is imported inside
every child environment: a stray `import ngio` there would require every peer's
environment to also hold ngio.

## Relationship to `tests/performance/`

The ngio repo has the other half of its performance work: a test that asserts
exact store-operation counts against committed baselines, answering *did this
change make ngio do more work?* as a pass/fail gate in CI. It stays there, since
it gates that repo's changes. This project answers what counts structurally
cannot: how it behaves at scale, whether it fits in memory, and how it compares.
