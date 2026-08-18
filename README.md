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

The five blocks, and the decision each one informs:

| block | axes | question |
| --- | --- | --- |
| `consolidate` | `mode` × `z` | which pyramid mode should I use, and will it fit? |
| `layout` | `layout` × `z` | same bytes, different chunk/shard shape |
| `roi` | `alignment` × `size` | chunk-aligned vs straddling reads |
| `algorithms` | `kernel` × `n` | scaling curves for ngio's own algorithms |
| `iterators` | `iterator` × `work` × `mapper` × `workers` | does fanning a map out actually pay? |

`algorithms` reports a series so the *shape* is visible — one timing cannot tell
O(n) from O(n²), four can. Its three kernels have different useful `n` ranges, so
cases outside a kernel's range are skipped and the run reports how many it
dropped; a bounded sweep must never read as a complete one.

`iterators` exists because the other half of ngio's performance work structurally
cannot answer this one. `tests/performance/` gates the iterators on **op counts**,
and it pins the parallel scenario's tally *equal* to the serial one — store
operations are invariant to concurrency by design there. Wall clock and peak
memory are the only instruments that can price a mapper, and both are here.

Its two arms are chosen so the pair tells the whole story. `features` reduces
read-only, so its curve is what the pool is worth on its own. `segmentation`
writes, and every writing map ends in a rebuild of the *whole* pyramid that no
pool touches — so its curve is the same speedup with a fixed serial tail bolted
on, and the gap between the two is Amdahl measured rather than asserted.
`consolidate` prices that tail on its own, which is what lets the two compose.

The `work` axis is there because a thread pool can only overlap what lets go of
the GIL, so a sweep with a trivial `func` measures the machinery's ceiling and
nothing a caller would meet. Its two values bracket the question with real
analysis code: `otsu` is a numpy histogram scan, which in isolation holds the GIL
outright (0.80 → 0.95 ms per patch across eight threads); `label` is
`scipy.ndimage.label`, whose C kernel releases it (1.30 → 0.22 ms, 5.9x). A third
value, `stub`, is the old near-free threshold, kept for isolating the machinery
and left out of the default because it flatters every mapper equally.

The measured result is not the obvious prediction from those two numbers, which is
why it is an axis rather than a pinned choice. **A GIL-bound `func` does not
flatten the pool** — on an 11-core laptop at eight threads, read-only: `otsu`
2.98x, `label` 3.46x; writing: 1.99x and 2.16x. Neither function is the majority
of a run at this geometry, so most of what the pool overlaps is the codec either
way, and the GIL dampens the speedup without deciding it. Read the pair as cost
against scalability: `label` is the more expensive function both serially (390 vs
281 ms) and at eight threads (113 vs 94 ms) — it never overtakes — but its higher
ratio buys back more of its own extra cost. `process` loses everywhere by 4–8x,
dominated by ~1 s per worker spawning and importing ngio in the child.

There is no such thing as a one-worker pool to measure: both parallel mappers
short-circuit to `BasicMapper` when the pool resolves to one, so `basic`,
`threaded@1` and `process@1` are one code path. The first sweep of this block
returned exactly that — three rows agreeing to within noise — and `basic` is now
the single serial row, with pools starting at two. A pool also never exceeds the
unit count, so a `workers` wider than the ROIs resolves down silently; the note
says so on any row where it binds, because a flat tail otherwise reads as "more
workers stop helping" when the pool simply stopped growing.

`peak_mb` is `tracemalloc`, and it sees nothing a worker *process* allocates —
the `process` rows report a fraction of a megabyte while doing all the work in
children. The suite has no `NATIVE = True` to suppress the column with, so those
rows carry the caveat in their note instead.

**This block needs an ngio newer than the `internal` extra installs.** None of
`ThreadedMapper`, `ProcessMapper` or `reduce_as_*` exists in 1.0.0, so in the
default interpreter it degrades to a single `unavailable` row and the other four
blocks run untouched. Point `[[environments]]` at a checkout to measure it — a
`—` column beside a numbered one reads correctly as *parallel mappers are new*.

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
which is a claim, and a wrong one. The same applies to a pure-Python library the
moment its chunks are encoded by the `zarrs` pipeline, so `pipeline = "zarrs"`
prints `n/a` too: the buffers left tracemalloc's reach whoever called them.

**`proc_peak_mb` — peak RSS of the process.** The OS high-water mark, so it
counts every byte including native buffers. It only ever rises, so a delta
around a single call is exact the first time and reads ~0 afterwards: measured
five times on the same 128 MiB read, the first call shows 133.5 MB and every
later one shows 0.1 MB.

That is why the comparison suites spawn **one child process per case** rather
than one per implementation. A shared child can only report the largest case it
ran and then attribute that number to all of them, which with `method` and
`[options.<impl>]` in the matrix is most of the rows. One process per case costs
an interpreter start — the environment is already installed and cached — and
buys a figure that is about the case it is printed next to. It also stops a case
inheriting the heap, the allocator's free lists and the warmed imports of the
one before it.

**`rss_base_mb` — the same mark, taken before the case ran.** Read after this
environment's imports and after the input array was materialised, but before the
first timed run. So `proc_peak_mb - rss_base_mb` is what the case cost, and
`rss_base_mb` on its own is the price of admission for that library — which for
several of them is the larger number. It is printed under the tables, because a
`peak RSS` column read without it ranks libraries by how much they import.

The `internal` suite still reports `proc_peak_mb` per process: its blocks share
fixtures and an interpreter, and `peak_mb` is the per-case column there.

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

`[[environments]]` is this suite's alone — `compare-io` and `compare-create`
reject it rather than silently ignoring it, because a matrix of ngio builds
is not what either suite reports: they compare `impl`s, and ngio is one row
among several. Comparing ngio versions there is `[env.ngio] requires = [...]`
(below) run once per build, with both runs appending to the same `csv` and the
recorded `ngio_version` column telling the rows apart —
`experiments/write-lock-v1-vs-worktree.toml` is a worked example, built to
catch exactly the kind of write-path regression `internal`'s read-mostly
blocks cannot see.

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
| `ngio` | `open_image` / `get_as_numpy` / `set_array`, via public slicing keywords; `mode = "dask"` swaps in `get_as_dask` and a dask patch |
| `zarr` | zarr-python directly — the floor every other row is read against |
| `zarrs` | the same code under the zarrs (Rust) codec pipeline; **zarr v3 only** |
| `dask` | `da.from_array` / `da.to_zarr`, region-aware, with `.compute()` inside the timing |
| `tensorstore` | the same bytes decoded outside Python — a ceiling, not a peer |
| `z5py` | a C++ zarr/n5 implementation with an h5py-shaped API |

The aligned/straddling pair is the read-amplification question asked of
everyone: a region inside chunk boundaries touches the minimum number of chunks,
and the same region offset by half a chunk fetches and discards every chunk on
each edge. The ratio between the two columns is what that costs.

`ngio`'s `mode` and the `dask` row invite a direct comparison on writes as well
as reads, which was not always a fair one to make. `dask`'s write used to go
through `da.store(..., lock=False)`, with the patch chunked to the target's
on-disk chunk shape rather than its **write unit** — the chunk, or the shard
when the array is sharded. zarr read-modify-writes a whole write unit whenever
a block does not cover one, so on `sharded` every block was a fraction of the
unit, and unlocked let two blocks racing on the same one lose each other's
updates: measured in
[`reports/dask-sharded-write-races.md`](reports/dask-sharded-write-races.md),
87% of a `sharded` `write_full` came out wrong, at 4.6× ngio's speed, in a
store one fifth the size.

That made `dask` the wrong floor to read ngio against, and
[`reports/ngio-upstream-write-path.md`](reports/ngio-upstream-write-path.md)
is why: since dask 2025.11, `da.to_zarr` onto an existing array asks the
target for its write unit, rechunks the patch to a multiple of it, and only
then stores — region-aware, correctly, with no lock, because the blocks it
hands to zarr are never smaller than what zarr has to read-modify-write.
`dask`'s write goes through `to_zarr` now, so the row measures the honest cost
of writing this correctly with no ngio layer on top, on any layout. The gap to
`ngio`'s row is then ngio's own abstraction cost — plus, on whichever ngio
build still locks every `da.store` call regardless of whether the block it is
about to write needs it, the cost of buying that same correctness the slow
way. The ngio-facing half of that finding — where rechunking the patch to the
shard shape turns 10.7 s into 0.58 s — is split out for filing upstream in
[`reports/ngio-dask-sharded-writes.md`](reports/ngio-dask-sharded-writes.md).

What to write *instead* is a third report,
[`reports/lazy-array-write-alternatives.md`](reports/lazy-array-write-alternatives.md):
ten strategies for getting a lazy array into a store, each with a runnable
snippet and its measured cost, over the three shapes ngio's callers actually
have — materialize, fan-in reduce, resample with a halo. Two results there bear
on this table. Rechunking to the write unit is *not* sufficient once the lock
comes off: on a straddling region it corrupts too, and the fix is to split the
region into its unit-aligned interior and the leftover faces. And
`_pyramid.py:47` rechunks to `target.chunks` rather than `target.shards or
target.chunks`, so `consolidate()` on a sharded pyramid carries the same 84×
amplification at every level.

That row is why `checksum` covers writes as well as reads. A read is hashed from
what it returned; a write is read back off disk afterwards by the `audit` hook,
with zarr-python rather than with whichever library wrote it — the auditor must
not be a contestant, or the one writer it can never catch is the one it agrees
with. A blank cell is a write nothing could read back, which is *unchecked*, not
passed.

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
`impl` × `image` × `method` × `pipeline` × each writer's own `[options]`.

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

### The filter has to be pinned too

The pyramid *shape* being pinned is not enough. Left to themselves these writers
also use four different filters, and the table used to record that as a
free-text string next to the timing:

| writer | what it really does when not told |
| --- | --- |
| `ngff-zarr` | itkwasm gaussian, through a WebAssembly build of ITK |
| `ome-zarr-py` | anti-aliased order-1 resize |
| `bioio` | `resize(order=0)` — nearest, no anti-aliasing |
| `ngio` | dask zoom, order 1 |
| `iohub`, `acquire-zarr` | 2×2 box mean |

Comparing those is a comparison of filters wearing the libraries' names. So
`method` is an axis with a canonical vocabulary each adapter translates:

```toml
method = ["mean", "nearest"]        # like-for-like across all six
```

| `method` | ngio | ngff-zarr | ome-zarr-py | bioio | iohub | acquire-zarr |
| --- | --- | --- | --- | --- | --- | --- |
| `default` | dask/linear | itkwasm gaussian | resize | native | mean | MEAN |
| `mean` | coarsen/linear | itkwasm bin-shrink | local_mean | — | mean | MEAN |
| `nearest` | numpy/nearest | dask-image nearest | nearest | native | stride | DECIMATE |
| `linear` | dask/linear | — | resize | — | — | — |
| `gaussian` | — | itkwasm gaussian | — | — | — | — |

A value a writer has no equivalent for is `unsupported` for that row, never
served by its default — a writer answering a request for `nearest` with a
gaussian would be the fastest kind of wrong. `default` stays the default value,
because "what does this library do untold" is the question the suite started
from. `method_native` records what each library was actually handed.

`ome-zarr-py` is the reason this is a declaration and not a `try`: 0.18 accepts
`gaussian` through the deprecated `Scaler` and *silently downgrades it to
resize*, so a row labelled gaussian would not have been one.

### The codec pipeline is an axis, not a seventh writer

`zarrs` is the Rust implementation of zarr's codec pipeline. It is not a library
with an API of its own — it installs a *replacement* for zarr-python's pipeline
through `zarr.config`, so whichever library is writing picks it up without
knowing it exists. That makes it a modifier on four of the six columns rather
than a seventh column:

```toml
pipeline = ["zarr-python", "zarrs"]     # every writer that goes through zarr-python
```

The delta between a writer's two rows is the pipeline and nothing else — same
library, same call, same store. On the smoke image:

| writer | `zarr-python` | `zarrs` | cpu/wall |
| --- | --- | --- | --- |
| `ngio` | 144.0 ms | 146.3 ms | 2.03 → 2.06 — *see below, zarrs never engaged* |
| `ome-zarr-py` | 212.3 ms | 196.8 ms | 1.50 → 1.44 |
| `bioio` | 108.0 ms | 66.4 ms | 1.10 → 1.37 |
| `ngff-zarr` | 4228.6 ms | 1121.4 ms | 1.21 → 1.79 |

The gains are smaller here than the ones `zarrs` publishes, and that is the
suite working rather than failing. Those are raw array reads and writes — this
one's `compare-io` measures the same shape and lands in the same ballpark (3.9×
on `read_full`, 2.3× on `write_full`). Building a pyramid also downsamples and
writes metadata, and the pipeline touches none of that, so the share it can move
is whatever fraction of the wall clock was encoding. `cpu/wall` is where to see
it: the writers that sped up are the ones that gained cores, because what zarrs
brings is a Rust encode parallel across chunks.

Three columns sit it out, each for a different reason and each saying which in
its `unsupported` note: `acquire-zarr` has no zarr-python anywhere in its path;
`iohub` picks its zarr stack through its own registry, which is
`[options.iohub] implementation` below; and any zarr **v2** image, because
`zarrs` implements the v3 codec pipeline and v2 has nothing to replace — a v2
row under it would silently measure zarr-python twice.

**The fallback is silent, and the audit columns do not catch it.** `zarrs`
installs its pipeline only for stores it recognises — `LocalStore`,
`ObjectStore`, `FsspecStore` — and for anything else zarr-python quietly keeps
`BatchedCodecPipeline`, with no warning and no error. The store it wrote is
byte-identical either way, so `codec`, `chunks`, `store` and `pyramid` all match
across the pair whether or not the swap happened. The tell is in the timing
itself: **wall *and* CPU seconds both unchanged** means the same code ran.

`ngio` is the case in point, and the reason its two rows above are within noise.
It wraps every store in `NgioStore`, a `zarr.storage.WrapperStore` subclass
carrying its retry logic, and there is no way to pass a raw `LocalStore` through
`create_empty_ome_zarr` — a `Path`, a `str` and a `LocalStore` all arrive
wrapped. zarrs therefore never engages, and ngio's `pipeline=zarrs` row is
zarr-python wearing a zarrs label. That is a fact about the store wrapper rather
than about ngio's writing, and it is the one row in this table to read with the
mechanism in mind.

`peak RAM` is blank on those rows for the same reason it is blank for
`acquire-zarr`: the chunks are encoded in Rust buffers `tracemalloc` cannot see,
and a `0.0` would read as "uses no memory" rather than "not measurable here".
`peak RSS` still applies.

Not swept by default, unlike `method`: it doubles every column at once, and a
config that did not name it installs and measures exactly what it did before the
axis existed. `compare-io` asks the same question as a pair of implementations
(`zarr` against `zarrs`) rather than as an axis, because there the measured code
is one module either way; `compare._pipeline` holds what both share.

### Each writer's own settings

Knobs with no cross-library equivalent are declared per adapter and swept from
an `[options.<impl>]` table. They only affect that writer's rows, and nothing
here sweeps by default:

```toml
[options.ngio]
mode = ["dask", "numpy", "coarsen"]              # machinery, not filter

[options.ngff-zarr]
use_tensorstore = [false, true]                  # a different write backend

[options.iohub]
implementation = ["zarr-python", "zarrs-python", "tensorstore"]

[options.bioio]
writer = ["full_volume", "timepoints"]

[options.acquire-zarr]
max_threads = [1, 8]
```

`[options.iohub] implementation` earns its place immediately, and is what the
`pipeline` axis above generalised. iohub's registry default is `zarrs-python` —
the Rust codec pipeline — while every other column encodes through zarr-python's.
On the smoke image that is 212 ms against 114 ms: roughly half of iohub's
headline advantage was a measurement of `zarrs`, inside a column labelled with
iohub's name. Having found that in one column, the axis asks it of the rest.

iohub keeps the option and declines the axis, alone among the zarr-python
writers: it is the same question in two vocabularies, and
`implementation=zarr-python pipeline=zarrs` is a row that reads as neither.

`method_native` is the escape hatch for filters with no canonical name (iohub's
`median`/`min`/`max`/`mode`, ngff-zarr's remaining `Methods`). It is mutually
exclusive with `method`; the row is labelled with whichever was used.

`compare-io` reads the same table. It has one option, which picks which of ngio's
two array types every operation goes through:

```toml
[options.ngio]
mode = ["numpy", "dask"]              # get_as_numpy, or get_as_dask + .compute()
```

### The audit columns

`levels`, `level_shapes`, `pyramid`, `codec`, `chunks` and `shards` are read
back **off disk** with `json` only, not taken from what the writer was asked for
— asking it would not catch a writer that did something else, and the auditor
must not be one of the contestants. `pyramid` says `as asked` or spells out the
divergence, so the one writer that cannot comply is labelled rather than
silently different.

This has already earned its place three times. It caught `acquire-zarr` reading
`max_levels` as the count of levels *below* level 0. It is how the
xy-versus-xyz split above was found — before pinning, it showed only as
`ngff-zarr` producing a 3.9 MB store where ngio produced 7.2 MB, which is easy
to misread as a codec difference. And `codec` is how the remaining spread turned
out to be exactly that: see below.

When the arrays on disk and the NGFF metadata disagree, both numbers are
reported: a store with five arrays declaring three is a different fault from
writing three.

## What these numbers do and do not measure

Every benchmark makes choices that a reader has to know about to use it. These
are the ones this suite makes.

**The centre is a median, and the width is beside it.** `seconds` is the median
of `repeats` timed runs; `seconds_mad`, `seconds_min` and `seconds_max` are the
rest of the distribution, and `repeats` says how many runs it came from. The
`spread` column prints `± <MAD>`, with a relative percentage once it passes 5%.
At one repeat it prints **`n=1`**, not `± 0.0`: a single sample of a pyramid
build on a laptop is not a precise number, and typesetting it as one is how a
comparison table lies. Median and MAD rather than mean and standard deviation
because at `repeats = 3` one interrupted run moves a mean by more than most of
the differences being looked for.

**Setup is outside the timing, and so is the previous run's store.** Every
`Measured` may carry a `setup` that runs before each execution and is excluded
from it. The create adapters use it to empty the target. Without that, run 1
writes into an empty directory and runs 2..n each begin by disposing of the
previous store — `zarr.open_group(mode="w")`, `overwrite=True`, an `rmtree` of a
few thousand files, depending on the library. That cost sat inside the timed
region for every run but the first, so the number both included a deletion
nobody asked about and depended on `repeats`.

A `gc.collect()` joins it there. Collection stays *enabled* during the run:
disabling it would flatter exactly the implementations that allocate most.

**Wall-clock, with CPU time beside it.** `seconds` is elapsed time, which is
what a caller waits for. But several of these writers use a dask scheduler or a
threaded codec pipeline, so elapsed time is partly a measurement of how many
cores the machine has. `cpu_seconds` is the CPU time over the same runs, and the
table shows a `cpu/wall` column whenever some row diverges from 1.0 by more than
20%. ngio's dask mode runs at about 2.0; iohub's zarr-python path at 0.88,
because it is waiting on I/O.

**Timing ends before the bytes are durable.** The measured call returns when the
library returns, with data still in the OS write-back cache. `sync_seconds`
times a flush after the timed runs and records it separately, rather than
folding it into `seconds` where it would distort the headline. How much work is
outstanding differs per writer, because their stores differ in size — so read
`seconds` as "what the call costs", not as write throughput.

**Reads are warm.** `compare-io`'s fixture is written once and read repeatedly,
and a read op gets one extra untimed call to compute its checksum. Nothing here
drops the page cache; there is no portable way to. So the read numbers measure
decode and copy, not disk.

**The codec has to be pinned, or the store size is not a result.** `compressors`
on an image spec defaults to `"auto"`, which is not one setting but six —
bioio's `blosc/zstd/3`, iohub's `blosc/zstd/1`, zarr-python's own default, and
acquire-zarr writing uncompressed. On identical pixels that produced stores
between 4.2 and 10.0 MB, a spread the README used to attribute vaguely to
"chunking and codecs". Pinned to `zstd`, the four writers that take a codec
object land within 0.15% of each other:

| | `auto` | `compressors = "zstd"` |
| --- | --- | --- |
| ngio | 7.18 MB | 7.18 MB |
| ngff-zarr | 7.18 MB | 7.18 MB |
| ome-zarr-py | 7.18 MB | 7.18 MB |
| bioio | **4.19 MB** | 7.18 MB |

iohub and acquire-zarr are blosc-only by API, so they land near but not on it;
the `codec` column is read off the store and says which is which.

**Machine state is recorded, not controlled.** There is no CPU pinning, no
turbo or frequency control, and no check that the machine is idle. `loadavg` is
recorded per case so an anomalous row can be attributed rather than puzzled
over. Results are gitignored for this reason — the recipe is the committable
artefact.

**What is still not pinned.** Thread counts, except acquire-zarr's `max_threads`
— the other libraries either expose none or expose them only through a dask
scheduler. `ngff-zarr.config.memory_target` *is* pinned, because left alone it
defaults to half of psutil-reported free RAM and silently chooses between a
single-pass and a slab-by-slab write path, which would make the same config
measure different code on different machines.

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

**Why not pixi.** The sibling `zarr-performance-exploration` sandbox uses it,
for a different shape of problem: a small, fixed set of named environments
declared once in a manifest. This project's environments are the opposite —
arbitrary, unbounded, and declared per committed experiment file, which is
what `[[environments]]` and `[env.<impl>]` above both are. `uv run --with` /
`--with-editable` installs whatever a config names with no manifest to edit;
pixi's environments are pre-declared in `pixi.toml` and solved by `pixi
install`, so a new comparison would mean editing that manifest and re-solving
before the config file could run at all — the opposite of "the config file is
the only interface".

### A blank cell is a claim

In a comparison table an unexplained gap reads as a result, so the reasons a cell
has no number stay distinguishable:

| cell | means |
| --- | --- |
| `unsupported` | the adapter declares it cannot express this case |
| `unavailable` | the environment failed to install, or the import failed |
| `failed` | it ran and raised; the CSV `note` has the exception |

`SUPPORTS`, `FORMATS` and `METHODS` on each adapter are declarations, not
documentation: `--list` prints every excluded case with its reason before
anything installs.

## Output

The comparison table is **long, not wide** — one row per implementation and
variant, columns are measurements:

```
create_pyramid   image=small   repeats=3
  impl          variant                                     median      spread   peak RAM   peak RSS      store  cpu/wall  lvl    pyramid     codec          vs first
  ngio          method=default mode=dask                  144.2 ms    ± 0.1 ms     5.0 MB   230.1 MB    6.8 MiB     2.02x  3      as asked    zstd              1.00x
  ngio          method=default mode=numpy                  75.8 ms    ± 0.3 ms    12.2 MB   234.4 MB    6.8 MiB     2.48x  3      as asked    zstd              0.53x
  bioio         method=default writer=full_volume         107.0 ms    ± 0.5 ms    10.7 MB   235.4 MB    4.0 MiB     1.10x  3      as asked    blosc/zstd/3      0.74x
  bioio         method=default writer=timepoints                                                   unsupported  bioio's write_timepoints needs a t axis
  iohub         method=default implementation=zarr-python 212.0 ms    ± 0.2 ms     2.5 MB   141.8 MB    5.5 MiB     0.88x  3      as asked    blosc/zstd/1      1.47x
  iohub         method=default implementation=zarrs-python 111.0 ms   ± 0.8 ms     0.5 MB   145.1 MB    5.5 MiB     1.36x  3      as asked    blosc/zstd/1      0.77x
```

It used to be wide — one column per implementation — which reads beautifully
right up to the point where the implementations stop having the same cases. With
`method` and `[options.<impl>]` they do not: ngio's `mode=numpy` has no
counterpart column in ngff-zarr, so a wide table spends most of its width on
`—`. Grouped by operation and then by image, so everything in one block is
directly comparable and the trailing ratio means something. That ratio is per
row against the first row that produced a number; the old table printed a `vs
<base>` heading over a value that was `last/base`, so with six columns four of
them got no ratio at all.

`variant` disappears when nothing varies, which is every row of a `compare-io`
file that sweeps no options.
The `cpu/wall` column appears only when some row diverges from 1.0 by more than
20%. `n/a` in `peak RAM` means `tracemalloc` could not account for that
implementation, not that it allocated nothing.

One CSV schema per suite, not a union across all three — the suites are separated
precisely so they are not read together. Within a suite the environment is
repeated on every row, so a single file answers cross-environment questions with
no joins, and runs append so several accumulate. A file whose header does not
match **raises** rather than being appended to — which it will for any results
file written before the measurement columns were added.

Axis columns hold **labels**, not values: `layout` is the string `sharded`,
`image` is `medium`, never a repr of a kwargs dict. A CSV cell and a config token
are the same token, so `df.pivot(index="image", columns="impl", values="seconds")`
is a one-liner.

Results are gitignored (`experiments/**/*.csv`) while the toml is not. The recipe
is the committable artefact; the numbers are machine-dependent.

### The HTML report

Past a couple of dozen rows the table stops being how anyone reads a run.
`reference-compare-create.csv` is 88 rows across six writers, two images and
four filters, a third of them `unsupported`; a `compare-io` sweep is six
operations deep across seven libraries. So each suite has a second reader:

```
uv run ngio-bench-report-internal       experiments/reference-internal.csv
uv run ngio-bench-report-compare-io     experiments/reference-compare-io.csv
uv run ngio-bench-report-compare-create experiments/reference-compare-create.csv
```

Each writes the CSV's path with `.html` — one self-contained file, no network,
no dependencies beyond what the runner already needs. Pass `-o` for a different
path or `--open` to launch it. They are gitignored for the same reason the CSVs
are.

One command per suite rather than one that sniffs the header, for the same
reason there is one runner per suite: each command's `check_csv` refuses the
other two at the door rather than half-rendering one of them. Underneath they
are one engine and three profiles (`ngio_benchmarks/report/_profile.py`), so
the three pages read as one family and a fix to any of it reaches all three.

Three views, one filter row scoping all of them:

* **Timing** — median wall-clock, faceted and grouped, with min/max whiskers
  where `repeats > 1`. Linear bars or a log dot plot; absolute or a ratio
  against a baseline you pick.
* **Coverage** — series against group, showing all four of `ok`,
  `unsupported`, `unavailable` and `failed`, with the adapter's own reason on
  hover. This is where the capability gaps read at a glance.
* **Memory & CPU** — for the comparison suites, peak RSS split into what
  importing the library cost and what the case cost on top; for `internal`,
  which takes no per-case baseline, tracemalloc's own figure. Both carry
  `cpu/wall` against a single-threaded rule.

**The audit column is drawn on the charts, not left in a column.** For
`compare-create` that is `pyramid`: a bar that built something other than what
was asked for is hatched and marked `≠`, because a writer that finished first
while writing something else has not won anything — `acquire-zarr` is the
fastest row in the reference file and every one of its bars carries that mark.
For `compare-io` it is `checksum`: every read in one cell opens the same store,
so a digest the rest of the cell did not agree on means that row read different
bytes, and it is hatched the same way. Majority rather than "matches ngio", so
ngio being the odd one out is reportable; writes carry no checksum and are
never marked. `internal` has no audit column — it measures one library against
itself, so there is no claim that two bars built the same thing to check.

Each report takes the **schema**, not one file. It works out which columns a
CSV actually varies along and offers them as facet, group and series pickers,
so a `compare-create` config sweeping `max_threads` or `writer` charts without
a code change, and an axis with one value simply drops a level of nesting. The
defaults differ per suite because the questions do: `compare-io` facets by
operation, since a full-level read and a straddling ROI write differ by orders
of magnitude and one shared scale would render the fast ones as slivers;
`internal` facets by block and takes no group axis, because its four blocks
sweep disjoint axes and any global group would caption three cards in four with
a value they do not have.

Colour names the series — the implementation, or the environment — and nothing
else, assigned by name so filtering never repaints the survivors. Seven hues
cannot all stay separable under simulated dichromacy inside the lightness bands
both themes need, so colour is deliberately a supporting channel: every bar is
directly labelled, every chart has a table view, and a texture toggle carries
identity with 45°/135° hatching for readers who need it without hue. An
`internal` file with no `[[environments]]` has one series and therefore one
colour throughout, and the legend says so rather than printing a one-item key.

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
call always share state — a `Measured.setup`, when a block needs one, is for
work that must be repeated before *every* run rather than done once.

**A new comparison implementation** is one module in the relevant `adapters/`
directory and one line in that suite's `IMPLS`:

```python
NAME     = "tensorstore"
REQUIRES = ("tensorstore>=0.1.85", "numpy>=2")   # its uv environment
SUPPORTS = frozenset({"read_full", ...})         # operations it can express
FORMATS  = frozenset({2, 3})                     # zarr formats it handles
PYTHON   = None                                  # interpreter pin, if any
METHODS  = {"mean": ..., "nearest": ...}         # canonical name -> its own
OPTIONS  = {"mode": ["dask", "numpy"]}           # its own settings, as axes
PIPELINES = frozenset({"zarr-python"})           # opt out of the pipeline axis

def build(op, spec, root, *, method=..., **options) -> Measured: ...

def excludes_pipeline(pipeline, values) -> str:  # one combination it declines
    ...
```

`METHODS` and `OPTIONS` are optional; an adapter declaring neither — every one in
`compare-io` but `ngio` — is called as `build(op, spec, root)`, since the parent
only passes keywords an adapter declared. That is also why an option needs a
`build`-side default: a config that does not name it sends nothing.

`OPTIONS` follows the same open/closed declaration forms as a block's `AXES`, and
an adapter's option names go in its suite's `AXIS_FIELDS` — `compare/create/__init__.py`
for a writer, `compare/io/__init__.py` for a reader. Without that the option still
runs and still labels the row's `case` and `variant`, but it gets no column of its
own — `output.as_row` fills the axis columns from `schema.axis_fields` and nothing
else — so the report cannot offer it as a facet, group or series.

An axis that picks only *how*, not *what*, also belongs out of `comparison_fields`,
which defaults to `axis_fields` and decides which rows the audit column holds to a
common answer. `compare-io` narrows it to exclude `mode`: ngio's two values read
one store through two APIs, so they must be checked against the peers' digest
rather than each landing in a cell of one that trivially agrees with itself.

`PIPELINES` and `excludes_pipeline` are the two ways out of the `pipeline` axis,
and both are optional — a writer that goes through zarr-python declares neither
and takes the swap without a line of code, which is the point. `PIPELINES` is
for an unconditional gap (`acquire-zarr`, which has no zarr-python in its path);
`excludes_pipeline` is for a combination only the adapter can judge (ngff-zarr's
tensorstore backend). Both are answered by the parent, so the gap shows up in
`--list` and no child process is started to discover it. The reason string is
the cell someone reads, so it should say what is actually true of *that*
library — see `iohub`, whose reason is not that it cannot.

If a `METHODS` entry or an `OPTIONS` value needs a package the library does not
itself depend on — `dask-image` for ngff-zarr's nearest, `tensorstore` for its
other backend — it goes in `REQUIRES`. A declaration is a promise, and `REQUIRES`
is what keeps it.

The constants are read by the **parent**, which has none of these libraries
installed — so an adapter must import with nothing but the standard library and
numpy. Every `import tensorstore` belongs inside `build`, never at module scope,
and a `METHODS` table maps to its library's *values* rather than its enum
members for the same reason. The same discipline applies to
`ngio_benchmarks/core/`, which is imported inside every child environment: a
stray `import ngio` there would require every peer's environment to also hold
ngio.

## Relationship to `tests/performance/`

The ngio repo has the other half of its performance work: a test that asserts
exact store-operation counts against committed baselines, answering *did this
change make ngio do more work?* as a pass/fail gate in CI. It stays there, since
it gates that repo's changes. This project answers what counts structurally
cannot: how it behaves at scale, whether it fits in memory, and how it compares.
