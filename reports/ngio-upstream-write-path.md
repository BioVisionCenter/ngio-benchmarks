# ngio's dask write path: five upstream fixes, in order

**Status:** measured, with committed harnesses — `reports/lazy_write_strategies.py` here, and `zarr-performance-exploration` for the multi-process, real-caller and zarrs results
**Affects:** ngio 1.0.0 and the current `feat/peformance-optimizations` branch; separately, ngio 0.5.14, which both downstream callers resolve to
**Cost today:** **16× slower than necessary** on a sharded write — 3546 ms against 220 ms — plus read amplification the lock does not touch, and a lock that does not hold across processes
**Fix:** go back to `da.to_zarr`, and stop treating dask's `PerformanceWarning` as an error

---

## 0. This reverses a deliberate decision, and that is the point

ngio has already been round this loop, and the changelog records both halves.

**v0.5.10** (`CHANGELOG.md:180`):

> Replace `da.to_zarr` with `da.store(..., lock=False)` in pyramid writes
> (`_on_disk_dask_zoom`, `_on_disk_coarsen`) and region slice writes
> (`_ops_slices`). Dask >=2025.11's `to_zarr` re-derives chunks via
> `normalize_chunks(chunks="auto", ...)` and emits a `PerformanceWarning`
> (treated as error by ngio's filterwarnings) when the result is not a multiple
> of the target's chunks; `da.store` writes blocks 1:1.

**v1.0.0** (`CHANGELOG.md:120`):

> Dask writes could silently drop data. `da.store(..., lock=False)` let two
> blocks read-modify-write the same chunk — or shard, when the target is sharded
> — concurrently, losing one update. […] All `da.store` calls now share a lock;
> block compute stays parallel.

Read together: v0.5.10 escaped a warning and introduced a data-loss bug; v1.0.0
fixed the bug it had introduced. Both changes were reasonable given what was
known. The finding here is about the first one:

> **That `PerformanceWarning` was dask correctly reporting that the write would
> not be write-unit aligned — the exact hazard v1.0.0 then had to fix with a
> lock. `to_zarr` was not the problem. It was, and is, the fix.**

From dask 2025.11, `to_zarr` onto an **existing** `zarr.Array` routes through
`_write_dask_to_existing_zarr`, which asks `_get_zarr_write_chunks` for the write
unit:

```python
def _get_zarr_write_chunks(zarr_array) -> tuple[int, ...]:
    """For Zarr v3 arrays with sharding, returns the shard shape.
    For arrays without sharding, returns the chunk shape."""
    if hasattr(zarr_array, "shards") and zarr_array.shards is not None:
        return zarr_array.shards
    return zarr_array.chunks
```

then rechunks the dask array to a multiple of it, warns when it cannot, and only
then calls `arr.store(z, lock=False, regions=...)` — unlocked, and safe
*because* of the rechunk. That is the same rule `ngio/common/_locks.py:8-9`
states in prose. dask implements it; ngio reimplemented the symptom's mitigation
instead.

Everything below follows from that, and the items are **ordered** — item 5 is a
3.3× regression if applied before items 1–2.

### A scoping note on every speedup below

Every figure in this report comes from arrays chunked at **128 KiB–512 KiB**.
Production plates in this ecosystem chunk at one FOV — `(1,1,2160,2560)` uint16
= **10.55 MB**, 20–80× larger — and per-chunk Python overhead amortises away at
that size. [`chunk-and-shard-layout.md`](chunk-and-shard-layout.md) measures the
curve: the zarrs read speedup falls from 6.1× to 1.86× across a 64× chunk-size
sweep, while the *write* speedup stays roughly flat (4.15× → 3.57×).

What that does and does not qualify:

- **The correctness findings are chunk-size independent.** Whether a block
  covers a whole write unit is geometry, not scale. Items 1–3 — the corruption,
  the read amplification, the per-process lock — hold at any chunk size.
- **The speedups are not.** Items 1, 2 and 5 quote ratios measured in the
  small-chunk regime. Expect them to shrink on FOV-chunked data, most on reads
  and least on writes.
- **Item 5's ordering conclusion survives**, because the `store/locked`
  regression under zarrs comes from whole-shard read-modify-writes, which large
  chunks do not remove.

---

## 1. `set_slice_as_dask` should use `da.to_zarr` again

`src/ngio/io_pipes/_ops_slices.py:263` and `:272`.

### Evidence

32 MB `uint16`, `chunks=(1,1,128,128)` inside `shards=(1,4,1024,1024)`, full
write, 4 workers, median of 3. Source blocks at the array's chunks — which is
what every ngio caller hands over today.

| arm | wall | store reads | MiB read | wrong |
| --- | --- | --- | --- | --- |
| `da.store(lock=DASK_STORE_LOCK)` — **today** | 3546 ms | 1024 | 2694.7 | 0 |
| `da.store(lock=False)` — v0.5.10–0.5.14 | 1302 ms | 1024 | 678.6 | **12,451,840** |
| **`da.to_zarr`** | **220 ms** | **0** | **0** | **0** |
| hand-rolled `rechunk(shards or chunks)` then store | 212 ms | 0 | 0 | 0 |

Zero store reads is the guarantee, not the clean `wrong` column: no
read-modify-write means no lost update, on any machine at any worker count.

At full size — 128 MB, 8 workers — the same locked path measures **11151 ms and
10816 MiB read** (`lazy-array-write-alternatives.md`, this repo). The `to_zarr`
arm has not been run at that scale, so the two are not mixed in one table here.

The region case is where `to_zarr` earns it outright. A 512² region at offset
129 into a `(1,1,256,256)`-chunked array, same run:

| arm | wall | store reads | wrong |
| --- | --- | --- | --- |
| `da.store(lock=DASK_STORE_LOCK)` | 173 ms | 256 | 0 |
| hand-rolled `rechunk(unit)`, unlocked | 123 ms | 256 | **1,696,502** |
| aligned interior + serial boundary faces | 100 ms | 128 | 0 |
| **`da.to_zarr`** | **66 ms** | 128 | **0** |

Rechunking to the write unit is **not sufficient on its own**: when the region
does not *start* on a unit boundary the rechunked blocks land on a grid offset
from the target's, every block straddles two units, and it corrupts — faster
than before. `to_zarr` handles that case correctly and still beats a hand-written
interior/boundary decomposition.

### The patch

```python
if ax is None:
    da.to_zarr(patch, zarr_array, region=slice_tuple)
    return
...
    da.to_zarr(sub_patch, zarr_array, region=_sub_slice)
```

and the comment at `:256-262` goes with it.

### The warning must stop being fatal, and start being useful

This is what killed the idea in v0.5.10, so it needs an explicit answer rather
than a hope.

`dask.array.core.PerformanceWarning` fires *exactly* when a region write cannot
be made unit-aligned. That is real, actionable information — it tells a caller
their ROI grid is misaligned with their chunk grid, which is a thing they can
fix and which costs them read amplification until they do. `filterwarnings =
["error"]` in `pyproject.toml:188` turns it into a test failure, which is why it
read as noise.

Minimum change:

```toml
filterwarnings = [
    "error",
    "default::dask.array.core.PerformanceWarning",  # dask reporting a misaligned
                                                    # region write -- information for
                                                    # the caller, not a defect
    ...
]
```

Worth considering on top: catch it at the call site and re-emit as an
`NgioUserWarning` naming the ROI and the write unit. ngio knows the geometry;
dask does not know it is being called by ngio.

### Risk

Requires the dask floor bump (see §6). Correctness on an unaligned region is
unchanged and speed improves; what changes is that the user now hears about it.

### CHANGELOG entry

> - **Dask writes are up to 16× faster on sharded arrays, and no longer need a
>   process-wide lock.** `set_slice_as_dask` went back to `da.to_zarr`, which
>   since dask 2025.11 rechunks the patch to the target's write unit — the shard
>   when sharded, else the chunk — before storing. A full write of a sharded
>   array goes from **3546 ms and 2694 MiB read to 220 ms and zero reads**. The
>   `PerformanceWarning` dask raises on a region write it cannot align is now
>   surfaced rather than suppressed: it reports genuine read amplification the
>   caller can remove by aligning the ROI grid.

---

## 2. `_pyramid.py` rechunks to `target.chunks`, not the write unit

`src/ngio/common/_pyramid.py:47` and `:106`:

```python
target_array = target_array.rechunk(target.chunks)
```

On a sharded target `target.chunks` is not the write unit, so `consolidate()`
carries the same amplification as an unaligned `set_array` — in the one place
the write unit was believed to be handled.

### Evidence

Measured on ngio's own `image.consolidate()`, sharded, via runtime patch:

| | wall | store reads | MiB read |
| --- | --- | --- | --- |
| as shipped (`target.chunks`) | 1259 ms | 3205 | **259.8** |
| `target.shards or target.chunks` | 954 ms | 2565 | **38.2** |
| `mode="numpy"` for reference | 226 ms | 13 | 33.9 |

Independently, on a reimplementation of both paths: `dask/target-chunks`
3411 ms / 1024 reads / 1009.3 MiB against `dask/target-unit` 2904 ms / 0 / 0.

### The patch

```python
unit = target.shards or target.chunks
target_array = target_array.rechunk(unit)
```

in both `_on_disk_dask_zoom` and `_on_disk_coarsen`.

**If item 1 lands, this becomes moot**: both call sites become `da.to_zarr` and
the rechunk disappears entirely, because dask does it. Do not implement both.
The standalone version is listed because it is a one-word change that can ship
immediately, without the dask floor bump.

### Nit, while in the file

The comment at `:43-46` says "the rechunk above already fixes every block shape",
but the rechunk is on `:47`, below it.

### CHANGELOG entry

> - **Consolidating a sharded pyramid moved 7× more data than it needed to.**
>   `_on_disk_dask_zoom` and `_on_disk_coarsen` rechunked to `target.chunks`,
>   which is not the write unit when the target is sharded, so zarr
>   read-modify-wrote a whole shard per block. Rechunking to `target.shards or
>   target.chunks` takes a sharded `consolidate()` from **259.8 MiB read to
>   38.2 MiB**.

---

## 3. `DASK_STORE_LOCK`: what it covers, and what it does not

Not "remove the lock" — it fixed a real bug and the v1.0.0 entry describes it
accurately. Two precise limitations, one measured and one structural.

### It does not hold across processes

`SerializableLock` pickles by token and rebuilds from a **per-process** registry,
so a child process gets a fresh lock, not a handle on the parent's. Four
processes writing sub-shard regions into one shard, single-threaded children so
the process count is the only source of contention:

| writer | procs=1 | procs=4 |
| --- | --- | --- |
| plain `arr[region] = data` | 0 | 3,145,728 (75%) |
| `da.store(lock=False)` | 0 | 3,096,576 (74%) |
| **`da.store(lock=DASK_STORE_LOCK)`** | 0 | **3,080,192 (73%)** |
| `da.to_zarr` | 0 | 3,145,728 (75%) |

Two things to take from this. The lock buys nothing here, as designed — there is
nothing shared to serialise on. And **plain numpy loses just as much**, so this
was never a dask problem: the hazard is zarr's read-modify-write of a shard, and
any two writers touching one shard from different processes lose updates. No
library-side change fixes it.

The fix is caller-side. In the `aligned` configuration — each process owning
whole write units — **every writer including the unlocked one is clean at four
processes**. That is a property of the partitioning, not of the API.

This matters concretely because `fractal-tasks-core` executes tasks as parallel
processes.

### It is module-global

`src/ngio/common/_locks.py:16` is one `SerializableLock` shared by all four
`da.store` call sites, so two unrelated images serialise their flushes against
each other. One lock per target array preserves the invariant at finer
granularity. Worth doing even if item 1 makes the lock largely redundant,
because the documented claim should be accurate either way.

### The call-site comment contradicts the definition

`_locks.py:8-10` states the rule correctly. `_ops_slices.py:259-262` restates it
as:

> The shared lock serialises the flushes: **a region write** whose blocks only
> partially cover a chunk (or, for a sharded target, a shard) makes zarr
> read-modify-write it

The "region write" qualifier is the misleading half. On a sharded target a
**full** write is affected identically and is in fact the worst case — it touches
every shard — and it is the more common call. Anyone reading the call site
concludes `set_array` with no slicing kwargs is the safe fast path. It is the
slow one.

---

## 4. Iterator API gaps

In descending order of how much they block real work.

### 4a. Shape-changing operations cannot be expressed at all

`ImageProcessingIterator` on a z-reduction, with either shaper:

```
NgioValueError: Incompatible shapes for patch and slice.
- Patch shape: (1, 1, 256, 256)
- Zarr array shape: (1, 1, 1024, 1024)
- Slice tuple: (slice(None), slice(1, 1), slice(0, 256), slice(0, 256))
- Expected shape: (1, 1, 1024, 1024)[...] (1, 0, 256, 256)
```

The ROI's z index is carried onto an output whose z extent is 1, so every ROI
past the first produces the empty slice `slice(1, 1)`. Reproduced identically on
sharded and unsharded inputs, with `by_chunks()` and with `by_yx()`.

This is why `fractal-tasks-core`'s projection cannot use the iterators today, and
it is the gap that matters most — an output-unit-driven reduce is otherwise the
best-performing strategy measured for that shape (322 ms against 2667 ms for the
current `set_array(dask)` path, with memory bounded at one z-column per worker).

### 4b. `by_chunks()` follows `chunks`, never `shards`

`src/ngio/iterators/_rois_utils.py:117` reads `ref_image.chunks`. Measured on a
sharded image:

| shaper | ROIs | write units | pixels overlap | units overlap |
| --- | --- | --- | --- | --- |
| `by_chunks()` | **4096** | **16** | no | no |
| `by_chunks(overlap_xy=2)` | 5184 | 16 | yes | **yes** |
| `by_yx()` | 64 | 16 | no | no |

4096 ROIs against 16 write units is 256 partial writes per shard — the
pathological pattern, with the ROI machinery's overhead on top. A
`by_write_unit()`, or making `by_chunks()` consult `shards or chunks`, fixes it.

`check_if_chunks_overlap` (`_abstract_iterator.py:384`) has the same issue: it
passes `self.ref_image.chunks`, so a sharded image's overlap check runs at the
wrong granularity and can report "no overlap" for ROIs that share a shard.

### 4c. `overlap_xy` produces write-overlapping ROIs

It is the only halo the API offers, and `check_if_chunks_overlap()` returns
`True` for its output. A resample needs a **read** halo with a non-overlapping
**write** core; a write halo is two ROIs racing for one unit.

### 4d. `BasicMapper` is strictly sequential

`MapperProtocol` (`_mappers.py:17`) exists precisely as the extension point, and
`require_no_chunks_overlap()` already exists to gate a parallel implementation. A
thread-pool mapper is roughly fifteen lines. Measured elsewhere in this
investigation, per-unit threaded writing is the fastest correct strategy for both
zarr→zarr shapes.

---

## 5. The zarrs blocker — worth 3–5×, but only after 1 and 2

### The blocker is wider than previously reported

`ngio-zarrs-codec-pipeline.md` identified `NgioStore` being a `WrapperStore`.
Probing store types directly:

| store | pipeline actually used |
| --- | --- |
| plain `LocalStore` | `ZarrsCodecPipeline` |
| `LocalStore` **subclass** | `BatchedCodecPipeline` — silent fallback |
| `WrapperStore(LocalStore)` | `BatchedCodecPipeline` — silent fallback |
| `MemoryStore` | `BatchedCodecPipeline` — silent fallback |

**No warning is emitted in any fallback case.** So unwrapping `NgioStore` is
necessary but may not be sufficient — even a minimal subclass is refused — and
any benchmark that sets `codec_pipeline.path` without verifying the engaged
pipeline is measuring zarr-python twice. That is worth knowing before anyone
sizes this work from a quick experiment.

### The ordering, which is the actionable part

Sharded full write, plain `LocalStore`, median of 3:

| arm | zarr-python | zarrs | speedup |
| --- | --- | --- | --- |
| `numpy` | 207 ms | 50 ms | **4.15×** |
| `to_zarr` | 332 ms | 68 ms | **4.87×** |
| `rechunk(unit)` + store | 265 ms | 57 ms | **4.66×** |
| interior/boundary split | 242 ms | 58 ms | **4.19×** |
| **`da.store(lock=DASK_STORE_LOCK)` — today** | 3100 ms | **10312 ms** | **0.30×** |

Every aligned path gains 3–5×. **The path ngio ships today gets 3.3× slower**,
reproduced on the straddling region (832 → 1465 ms, 0.57×). zarrs accelerates
encode and decode; the locked path's cost *is* 1024 whole-shard
read-modify-writes, and pushing each of those through the Rust boundary costs
more than the faster codec saves. zarrs amplifies the amplification.

This also explains the flat result in the earlier report — `consolidate` under
zarrs measured 1394 → 1431 ms, "none". It was not that zarrs does not help; it
is that the write path being measured is the one zarrs cannot help.

### Proposal

A documented way to obtain the unwrapped store. `NgioStore` already exposes
`local_root` (`_store.py:172-177`), `store_type` and `is_local`, which fits its
stated design — "ask the store for the service it needs" — so an `unwrap()` or
`inner` property is the smallest addition consistent with it.

Framed as an opt-in, because ngio's retry policy lives in the wrapper and
unwrapping forfeits it. Whether that trade is acceptable, and under what flag,
is a maintainer call this report does not try to make.

---

## 6. Test and packaging impact

The part most likely to be underestimated.

### `tests/unit/io_pipes/test_dask_write_race.py` will fail, and correctly so

It writes `[4, 20)` into a `chunks=(16,)` array with 8-element dask blocks, so
two blocks partially cover chunk 0, and asserts:

```python
# Locked or not, the contested chunk is read once per block. Fewer means
# the chunk key drifted and the gate never fired, making the test vacuous.
assert gated.arrivals >= 2
```

Under `to_zarr`, dask rechunks the patch and the contention the test constructs
no longer exists, so the count drops below 2. The comment says fewer arrivals
means the test went vacuous; after the fix it means **the fix worked**.

Do not delete this test — it is the only deterministic repro of the race in the
suite. Keep `np.testing.assert_array_equal(array[:], reference[:])`, and replace
the arrivals assertion with one that pins the new, stronger guarantee: the
contested chunk is no longer read once per block. The test also needs the
`PerformanceWarning` filter from item 1, since its region is deliberately
misaligned and will trigger it.

### Other affected tests

- `tests/unit/common/test_pyramid.py::test_on_disk_zoom_sharded_matches_unsharded`
  names the shared lock in its docstring. The assertion should still hold; the
  docstring needs updating.
- `tests/performance/baselines/{local,memory}.json` need regenerating.
  `consolidate_dask`, `consolidate_coarsen` and `consolidate_numpy` record exact
  `get.chunk` and `bytes.read` counts, and items 1–2 are *designed* to move them.
  `pixi run -e test11 pytest tests/performance -p no:xdist --update-baseline`.
  Both files are currently staged-modified on the branch, so regenerate on a
  clean tree or the pending deltas fold in silently.
- `tests/performance/scenarios.py:218-227` is already stale — it describes the
  `compute_chunk_sizes()` double-read that a recent commit removed.

### `pyproject.toml`

```toml
"dask[array]>=2025.11",   # was >=2024.1
```

plus the `filterwarnings` line from item 1. The floor bump is what makes item 1
a single code path rather than a version-gated branch.

---

## 7. Cross-repo consequence

`fractal-tasks-core` pins `dask>=2023.1.0,<2025.11` — it explicitly excludes the
dask that added the alignment. **With the floor bump it cannot adopt the fixed
ngio until that pin is lifted.** That is a coordinated change across two repos,
not a footnote, and it is the main cost of choosing the floor bump over a
version-gated dual path.

Worth knowing while planning it: both `fractal-tasks-core` and
`ome-zarr-converters-tools` pin `ngio>=0.5.8,<0.6.0` and today resolve to **ngio
0.5.14**, whose write path is `da.store(..., lock=False)` with no `_locks.py` —
the window between v0.5.10 and v1.0.0, with neither dask's alignment nor the
lock.

Stated without overclaiming: measured on that exact stack, `projection()` shows
**exposure, not corruption**. A sharded source gives 2195 store reads over 11
keys — 77 MiB moved to write a 1.5 MiB store — and `wrong = 0`. A non-zero read
count is an exposure whose realisation depends on contention; only a zero read
count is a guarantee. The unsharded cases show no amplification at all, because
`abstract_derive` (`_abstract_image.py:1090-1093`) inherits both chunks and
shards from the source, so blocks and target grid match by construction unless
the source is sharded.

---

## 8. Suggested order

1. **Item 2 alone** — one word, no dependency, ships now.
2. **Item 1** with the dask floor bump and the warning filter. Supersedes item 2's
   rechunk; do not implement both.
3. **Item 3** documentation now; lock granularity or removal after item 1 settles.
4. **Item 4** — independent of the rest, and 4a is what unblocks a downstream
   caller.
5. **Item 5** last. Before items 1–2 it is a 3.3× regression on sharded writes.

---

## 9. Scope of the claim

zarr-python 3.3.0; ngio 1.0.0 and 0.5.14; dask 2026.7.1 and 2025.10.0; zarrs
0.2.3; tensorstore 0.1.85; numpy 2.5.2; Python 3.13.9; macOS 26.6.1 arm64.
`LocalStore` and `MemoryStore` only — the corruption reproduces identically on
`MemoryStore` with the same read counts, so it is not a filesystem artefact. Not
tested against a real object store, or on Linux.

**Measured:** every table above. Multi-process results are observed lost updates,
not inferences. The iterator gaps are reproduced failures with the exceptions
pasted. Item 2's numbers come from ngio's own `image.consolidate()`, not a
reimplementation.

**Qualified.** The item-1 and item-5 tables are `--quick` (32 MB) runs at
median-of-3; direction and magnitude are consistent across layouts and trial
counts, but a full-size run is worth doing before anyone quotes an exact figure,
and 32 MB and 128 MB numbers are never mixed in one table. Item 2's *wall times*
are single-shot — its load-bearing claim is the byte count, which is
deterministic. `wrong` is always worst-of-trials, never median: one corrupt
trial proves lost updates, a clean one proves nothing.

**Not measured:** real S3 latency; whether unwrapping `NgioStore` is sufficient
for zarrs beyond the store-type probe; the cost of lifting fractal's dask pin.

## Reproducing

```bash
# this repo
uv run python reports/lazy_write_strategies.py

# the sandbox, which carries the multi-process, real-caller and zarrs arms
cd ../zarr-performance-exploration
pixi run versions                 # which stack, and what protects it
pixi run strategies               # item 1
pixi run consolidate --patch pyramid-unit   # item 2
pixi run multiproc                # item 3
pixi run -e callers projection    # items 4a, 7
pixi run zarrs                    # item 5
```
