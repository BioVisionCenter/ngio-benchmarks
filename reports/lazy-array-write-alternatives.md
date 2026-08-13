# If not `da.store`, then what? Writing lazy arrays to zarr

**Status:** measured, with a committed harness — `reports/lazy_write_strategies.py`
**Scope:** what to use instead of `da.store` for the three shapes ngio's callers actually have
**Constraint:** the recommendation must be correct **without an in-process lock**, because Fractal runs tasks as parallel processes and `SerializableLock` is per-process

Two companion reports established what is wrong with `da.store`:
[`dask-sharded-write-races.md`](dask-sharded-write-races.md) (it silently loses
~87% of a sharded write) and
[`ngio-dask-sharded-writes.md`](ngio-dask-sharded-writes.md) (it costs 18× more
than it needs to). This one is the menu of replacements. Every section below is
a strategy, a runnable snippet, and its measured cost.

---

## 1. The rule

> **The unit of a safe parallel write is the shard — or the chunk when the array
> is unsharded. Never the dask block.**

Everything else here is a corollary. `da.store` issues one `zarr.Array.__setitem__`
per dask block; zarr skips the read only for a write covering a *whole* unit; and
two writes that read-modify-write the same unit concurrently lose one of the
updates. So there is exactly one question to ask of any write strategy:

> **Can two of its writes touch the same write unit?**

If no — it is safe at any worker count, in any process, with no lock. If yes — it
needs either serialisation or a different decomposition.

This predicate is easy to state wrongly, and has been, four times: three places
in this repo and once upstream, at `ngio/io_pipes/_ops_slices.py:257-262`, which
qualifies it as "a **region** write whose blocks only partially cover a chunk".
Both qualifiers mislead. On a sharded target a *full* write is affected
identically and is the worst case; and a block can cover its chunk exactly and
still be 1/256 of the shard that actually gets written.

### The correction this report adds

The fix proposed in `ngio-dask-sharded-writes.md` — rechunk the patch to
`shards or chunks` before `da.store` — is **necessary but not sufficient once
the lock is removed.** That report says so in passing ("a region write that does
not start on a unit boundary still cannot be made complete, so the lock must
stay"); this one puts a number on what happens if you take the rechunk and drop
the lock anyway, which is what the multi-process constraint forces:

| unsharded, 512² region at offset 129, 8 workers | wall | store reads | wrong elements |
| --- | --- | --- | --- |
| `store/locked` — ngio today | 547 ms | 1024 | 0 |
| `store/unit` — rechunk, no lock | 349 ms | 1024 | **7,174,890** (43%) |
| `split` — align, then peel the edges | **294 ms** | 512 | **0** |

Rechunking to the unit only helps when the region **starts** on a unit boundary.
Otherwise the rechunked blocks land on a grid offset from the target's, every
block straddles two units, and it corrupts exactly as before — faster.

---

## 2. Decision table

| strategy | lock-free | multi-process safe | memory bound | use when |
| --- | --- | --- | --- | --- |
| `numpy` | yes | yes | **whole region** | it fits in RAM; it is the ceiling |
| `store/locked` | no | **no** | one block × workers | nothing — this is today's default and it is dominated |
| `store/unlocked` | yes | **no** | one block × workers | never; the known-bad control |
| `store/unit` | yes | yes *only if the region start is unit-aligned* | one unit × workers | full-array writes, and region writes you can prove aligned |
| `split` | yes | interior yes; faces need caller coordination | one unit × workers | **the general-purpose answer** for arbitrary regions |
| `eager` | yes | yes | **one unit** | memory is the binding constraint |
| `eager/threads` | yes | yes | one unit × workers | zarr→zarr transforms; large write units |
| `eager/batched` | yes | yes | batch × unit | one source tile feeds several output units |
| `tensorstore` | yes | yes | its own | you can take the dependency — **fastest measured, everywhere** |
| `cubed` | yes | yes | one unit *(not exercised — see §4)* | you want unaligned regions to **raise** rather than corrupt |

---

## 3. The headline measurement

128 MB `uint16`, shape `(1, 64, 1024, 1024)`, full write, 8 workers, median of 3.
`sharded` is `chunks=(1,1,128,128)` inside `shards=(1,4,1024,1024)`; `unsharded`
is `chunks=(1,1,256,256)`. Source blocks = the array's chunks, which is what
every caller in ngio hands over today.

| strategy | unsharded wall | sharded wall | sharded reads | sharded MiB read | wrong |
| --- | --- | --- | --- | --- | --- |
| `numpy` | 373 ms | 614 ms | 0 | 0 | 0 |
| `store/locked` — **ngio today** | 1776 ms | **11151 ms** | 4096 | **10816.5** | 0 |
| `store/unlocked` | 643 ms | 3497 ms | 4096 | 1452.1 | **57,917,440** |
| `store/unit` | 606 ms | **703 ms** | **0** | **0** | 0 |
| `split` | 604 ms | **700 ms** | **0** | **0** | 0 |
| `eager` | 2068 ms | 1085 ms | 0 | 0 | 0 |
| `eager/threads` | 954 ms | 944 ms | 0 | 0 | 0 |
| `eager/batched` | 2006 ms | 1226 ms | 0 | 0 | 0 |
| `tensorstore` | **230 ms** | **168 ms** | n/a | n/a | 0 |
| `cubed` | 734 ms | 694 ms | 0 | 0 | 0 |

Three things to read off it.

**Zero store reads is the guarantee.** A clean `wrong` column proves nothing on
its own — absence of an observed race never does. Zero reads proves there was no
read-modify-write, and with no read there is nothing to lose, on any machine at
any worker count. Every recommended strategy below shows zero.

**Reading *fewer* bytes is the corruption signature.** `store/unlocked` reads
1452 MiB where the locked run reads 10816, and writes a store of 11.7 MiB where
every correct arm writes 84.8. It looks like the fast one and it looks like it
compressed better. It lost 86% of the array.

**The lock is not the expensive part; the amplification is.** `store/locked` and
a serial unlocked run both move 10816 MiB to write 128 MB — 84×. Rechunking to
the write unit takes that to zero, which is why `store/unit` is 16× faster than
`store/locked` while being *more* correct.

---

## 4. The strategies

Every strategy snippet below is the **verbatim** body of the corresponding
harness arm — checked, not asserted; regenerate them with
`uv run python reports/lazy_write_strategies.py --emit-snippets`. The four
shared helpers, shown once here, are the exception: their signatures and
docstrings are trimmed for reading, and the full versions are in the harness and
in `_probe.py`.

```python
def write_unit(array: zarr.Array) -> tuple[int, ...]:
    """The granularity zarr actually writes in: the shard, else the chunk."""
    return array.shards or array.chunks


def unit_grid(index, shape, unit):
    """The target's write-unit grid, clipped to `index`, in absolute coordinates.

    Yields one slice tuple per write unit the region touches, each already
    trimmed to the region. Two properties every caller depends on: the pieces
    are **disjoint**, so threads writing different ones cannot race; and there
    is at most **one per unit**, so a region that does not start on a unit
    boundary costs one read-modify-write of the edge units rather than one per
    dask block.
    """
    per_axis = []
    for (start, stop), size in zip(bounds(index, shape), unit, strict=True):
        edges = []
        edge = (start // size) * size
        while edge < stop:
            edges.append((max(edge, start), min(edge + size, stop)))
            edge += size
        per_axis.append(edges)
    for combo in itertools.product(*per_axis):
        yield tuple(slice(lo, hi) for lo, hi in combo)


def local(piece, index, shape):
    """An absolute slice tuple, rebased onto a patch that covers `index`."""
    return tuple(
        slice(sl.start - start, sl.stop - start)
        for sl, (start, _) in zip(piece, bounds(index, shape), strict=True)
    )


def threaded(work: Callable[[Any], None], items: Sequence[Any], workers: int) -> None:
    """Run `work` over `items`, in a pool or inline when `workers == 1`."""
    if workers == 1:
        for item in items:
            work(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, items))
```

`bounds(index, shape)` is `(start, stop)` per axis with `None`s resolved, and
`split_aligned(index, shape, unit)` returns the unit-aligned interior box and
the leftover faces; both are in the harness.

### `store/unit` — rechunk to the write unit, then store lock-free

One block per unit means every write covers a whole unit, so zarr skips the read
entirely and there is nothing left for a lock to protect. `rechunk` is a graph
operation: it costs nothing at call time.

```python
unit = write_unit(array)
if patch.chunksize != unit:
    patch = patch.rechunk(unit)
with dask.config.set(scheduler="threads", num_workers=workers):
    da.store(patch, array, regions=tuple(index), lock=False)
```

| | wall | reads | MiB read | wrong |
| --- | --- | --- | --- | --- |
| sharded, full | 703 ms | 0 | 0 | 0 |
| unsharded, full | 606 ms | 0 | 0 | 0 |
| unsharded, **straddling region** | 349 ms | 1024 | 0.2 | **7,174,890** |

**Costs.** Peak memory rises to one unit per in-flight task — 8 MiB per block
here against 32 KiB — though it replaces 256 decode-merge-re-encode cycles per
shard, so it allocates less than it removes. And the third row: **this is only
safe when the region starts on a unit boundary.** Use it for full-array writes;
for arbitrary regions use `split`.

### `split` — aligned interior in parallel, unaligned faces serially

The general-purpose answer. The interior is the largest box inside the region
whose every face lands on a unit boundary, so writing it can never read-modify-
write anything; the leftover is peeled one axis at a time into at most two faces
per axis, each at most one unit thick, written one at a time.

```python
unit = write_unit(array)
interior, faces = split_aligned(index, array.shape, unit)
if interior is not None:
    inner = patch[local(interior, index, array.shape)].rechunk(unit)
    with dask.config.set(scheduler="threads", num_workers=workers):
        da.store(inner, array, regions=tuple(interior), lock=False)
for face in faces:
    array[face] = patch[local(face, index, array.shape)].compute()
```

| | wall | reads | MiB read | wrong |
| --- | --- | --- | --- | --- |
| sharded, full | 700 ms | 0 | 0 | 0 |
| unsharded, full | 604 ms | 0 | 0 | 0 |
| unsharded, straddling region | **294 ms** | 512 | 0.0 | **0** |
| sharded, straddling region | 255 ms | 16 | 0.0 | 0 |

Faster than `store/locked` on the straddling region (294 ms against 547 ms)
*and* correct, where `store/unit` is fast and wrong.

**Costs.** One honest limitation: the interior is race-free across processes,
but the **faces still read-modify-write**. Two independent processes writing
regions whose faces land in the same unit can still lose an update. `split`
removes every race *within* one write and confines the cross-write risk to
boundary units; making that risk zero means aligning the region grid to the unit
grid, which is a caller-side decision (see the converter recommendation below).

### `eager` / `eager/threads` — compute per write unit, write complete units

Never build a dask block grid at all. Walk the target's own write units, compute
each one, write it whole. Lock-free by construction, because the units are
disjoint and there is exactly one write per unit.

```python
def write_one(piece: tuple[slice, ...]) -> None:
    array[piece] = patch[local(piece, index, array.shape)].compute(
        scheduler="synchronous"
    )

threaded(write_one, list(unit_grid(index, array.shape, write_unit(array))), workers)
```

Each unit computes single-threaded because the parallelism is the outer pool;
nesting dask's own pool inside it oversubscribes the machine. The sequential
`eager` arm is the same loop without `threaded`.

| | `eager` | `eager/threads` | reads |
| --- | --- | --- | --- |
| sharded, full (unit = 8 MiB, 16 units) | 1085 ms | 944 ms | 0 |
| unsharded, full (unit = 128 KiB, 1024 units) | 2068 ms | 954 ms | 0 |
| B1 reduce, sharded | — | **322 ms** vs 2667 ms for `store/locked` | 0 |
| B2 resample, sharded | — | **846 ms** vs 3411 ms as ngio ships | 0 |
| B2 resample, unsharded | — | 1735 ms vs 487 ms for `numpy` | 0 |

**Costs.** Memory is bounded at one unit per worker regardless of array size —
the only strategy here with that property, and the reason it wins the zarr→zarr
shapes. But per-unit overhead is fixed, so it **loses badly when the unit is
small**: 1024 tiny computes on the unsharded array cost more than the writing
saved. The rule is unit size, not array size — it wins on sharded targets and on
transforms, and loses on finely chunked unsharded ones.

### `eager/batched` — one graph per batch of units

Recovers what per-unit `.compute()` throws away: when one source tile feeds
several output units, a shared graph loads it once instead of once per unit.

```python
pieces = list(unit_grid(index, array.shape, write_unit(array)))
for start in range(0, len(pieces), batch):
    group = pieces[start : start + batch]
    blocks = dask.compute(
        *[patch[local(piece, index, array.shape)] for piece in group],
        scheduler="threads",
        num_workers=workers,
    )
    for piece, block in zip(group, blocks, strict=True):
        array[piece] = block
```

**Measured 2006 ms / 1226 ms — slower than plain `eager/threads` on this
harness, and it should not be taken as a win.** The synthetic patch is a
materialized numpy array, so there is no shared load for batching to
deduplicate; this arm only pays the barrier. It is listed because the
converter's real graph *does* have that sharing (one tile spanning several
chunks), and this is where to look if the eager arms disappoint there — not
because the number here recommends it.

### `tensorstore` — hand off the IO entirely

```python
handle = ts.open(
    {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(array.store.root)},
    },
    open=True,
).result()
handle[tuple(index)].write(patch.compute()).result()
```

| | wall | vs `numpy` (best native) | vs `store/locked` | wrong |
| --- | --- | --- | --- | --- |
| sharded, full | **168 ms** | 3.7× faster | **66× faster** | 0 |
| unsharded, full | 230 ms | 1.6× faster | 7.7× faster | 0 |
| sharded, straddling | 63 ms | 3.7× faster | 38× faster | 0 |

The fastest thing measured anywhere in this report — 1.6–3.7× faster than the
best native strategy on every arm — correct in all of them, and it does its own
chunk-aligned IO so the write-unit question never arises.

**Costs.** A C++ dependency, and its own metadata handling — the store is opened
by path, not through zarr-python, so ngio's `NgioStore` wrapper and anything
layered on it is bypassed. Its read counts read `n/a` above rather than `0`
because the counting store never sees them; that is a limit of the instrument,
not a measurement of zero.

### `cubed` — bounded-memory dask replacement

```python
source = cubed.from_array(patch.compute(), chunks=write_unit(array))
cubed.store([source], [array], regions=[tuple(index)])
```

694 ms sharded / 734 ms unsharded, zero reads, zero wrong — competitive with
`store/unit`.

**What this arm does *not* show is cubed's headline property.** `cubed.from_array`
needs a concrete array, so the arm calls `patch.compute()` first and the whole
region is resident before cubed sees it. Its bounded-memory execution model is
therefore untested here, and the timing is "cubed as a writer", not "cubed as a
dask replacement". Sizing that properly means building the pipeline in cubed
from the source upwards, which is a different investigation.

What it does show is one design decision worth the whole arm:

```
ValueError: Region (..., slice(129, 641), slice(129, 641)) does not align
with target chunks (1, 1, 256, 256)
```

**cubed refuses the unaligned region rather than corrupting it.** That is the
single most defensible behaviour in this whole comparison, and it is worth
copying regardless of whether cubed itself is adopted: the failure mode this
report exists to document is that the unsafe write returns cleanly.

### Rejected: `store/locked` and `store/unlocked`

`store/locked` is ngio today. It is correct in-process and dominated on every
axis: 11151 ms where `store/unit` takes 703 ms, 84× read amplification the lock
does not touch, and — decisively — `SerializableLock` is per-process, so it does
not make a Fractal task safe against a sibling task. `store/unlocked` is dask's
default and lost 57,917,440 of 67,108,864 elements (86%) on the sharded full
write. Neither is a candidate.

---

## 5. The zarr→zarr shapes

### B1 — fan-in reduce (projection / MIP)

`(1,64,1024,1024)` → `(1,1,1024,1024)`, max over z.

| strategy | unsharded | sharded | sharded reads | wrong |
| --- | --- | --- | --- | --- |
| `numpy` | 249 ms | 369 ms | 0 | 0 |
| `store/locked` — **fractal today** | 520 ms | 2667 ms | 64 | 0 |
| `store/unit` | 538 ms | 2617 ms | 0 | 0 |
| `eager/threads` | **202 ms** | **322 ms** | 0 | 0 |
| `iterators/by_chunks` | **fails** | **fails** | — | — |
| `iterators/by_yx` | **fails** | **fails** | — | — |

`eager/threads` is 8× faster than the current path on a sharded target, beats
even the whole-array numpy path, and is the only one of the four with bounded
memory — it holds one z-column per worker regardless of stack depth.

Note that `store/unit` is 2617 ms with **zero reads**. Removing the read
amplification is necessary but not sufficient: what is left is dask's own
rechunk-and-graph overhead, and it is the dominant cost here. That is the
strongest single argument for leaving dask out of the write path rather than
tuning it.

**Both `ngio.iterators` arms fail outright**, identically on sharded and
unsharded:

```
NgioValueError: Incompatible shapes for patch and slice.
- Patch shape: (1, 1, 256, 256)
- Zarr array shape: (1, 1, 1024, 1024)
- Slice tuple: (slice(None), slice(1, 1), slice(0, 256), slice(0, 256))
```

The ROI's z index is carried straight onto the output, which has `z == 1`, so
every ROI past the first produces the empty slice `slice(1, 1)`.
`ImageProcessingIterator` cannot express a shape-changing reduction today — with
`by_chunks()` or `by_yx()`. This is the API gap that matters most, because it is
the API this report otherwise recommends building on.

### B2 — resample with a halo (pyramid consolidate / zoom)

Half-scale in yx.

| strategy | unsharded | sharded | sharded reads | sharded MiB read |
| --- | --- | --- | --- | --- |
| `numpy` | 487 ms | 651 ms | 0 | 0 |
| `dask/target-chunks` — **ngio today** | 599 ms | **3411 ms** | **1024** | **1009.3** |
| `dask/target-unit` | 519 ms | 2904 ms | **0** | **0** |
| `coarsen/target-chunks` — ngio today | 616 ms | 3443 ms | 1024 | 1009.3 |
| `coarsen/target-unit` | 531 ms | 3001 ms | 0 | 0 |
| `eager/threads` | 1735 ms | **846 ms** | 0 | 0 |

**A finding no existing report covers.** `ngio/common/_pyramid.py:47` and `:106`
rechunk to `target.chunks`, not `target.shards or target.chunks`:

```python
target_array = target_array.rechunk(target.chunks)
da.store(target_array, target, lock=DASK_STORE_LOCK)
```

On a sharded pyramid that aligns the blocks to the wrong grid, so `consolidate()`
carries the *same* read amplification as `set_slice_as_dask` — 1024 whole-shard
reads over 16 keys, 1009 MiB moved to write 32 MiB — at every level. It is the
one place the write unit was thought to have been handled. Changing `.chunks` to
`.shards or .chunks` takes the reads to zero.

The `eager/threads` arm reads each output unit's source box plus a two-pixel
halo, zooms it, and writes the core. It is exact against a whole-array
`numpy_zoom` (`wrong = 0`), 4× faster than the shipped path on a sharded target,
and bounded in memory — but 3.5× *slower* than plain numpy on the unsharded one,
for the small-unit reason given above.

`coarsen` is scored against a coarsen reference, not a zoom one: a block mean and
a linear interpolation are both correct downsamplers that disagree by up to one
grey level, and scoring one against the other produced a five-figure `wrong`
count for an arm that is not wrong at all.

### What `ngio.iterators` produces today

| shaper | ROIs (sharded) | write units | pixels overlap | **units overlap** |
| --- | --- | --- | --- | --- |
| `by_chunks()` | **4096** | **16** | no | no |
| `by_chunks(overlap_xy=2)` | 5184 | 16 | yes | **yes** |
| `by_yx()` | 64 | 16 | no | no |

Two gaps, both directly relevant:

- **`by_chunks()` follows `chunks`, not the write unit.** On the sharded array
  it yields 4096 ROIs against 16 write units — 256 partial writes per shard,
  which is exactly the pathological pattern, now with the ROI machinery's
  overhead on top.
- **`by_chunks(overlap_xy=...)` overlaps on *write*.** It is the only halo the
  API offers, and `check_if_chunks_overlap()` returns `True` for it. A resample
  needs a **read** halo with a non-overlapping write core; a write halo is two
  ROIs racing for one unit.

---

## 6. Recommendations, per use case

### `ome-zarr-converters-tools/.../pipelines/_to_zarr.py`

**This caller is not exposed to the bug the rest of this report is about, and
the two dask modes are not the same.** Both `load_data_dask` overloads default
to `chunks = shape` (`core/_tile_region.py:149` and `:360`), so each one hands
`set_roi` a **single-block** dask array — but "shape" means something different
in each:

| mode | one block covers | peak resident | `__setitem__` calls per `set_roi` |
| --- | --- | --- | --- |
| `BY_TILE_DASK` | the **whole image** | whole image + its tiles | 1 |
| `BY_FOV_DASK` | **one FOV** | one FOV + its tiles | 1 |

Two consequences, and the first one corrects a natural assumption:

- **`BY_FOV_DASK` is not an in-memory mode.** It loops over FOV groups
  (`_to_zarr.py:89`) and builds a fresh single-block array per group, so its
  memory profile is the same as sequential `BY_FOV`. What dask buys it is
  **parallel tile loading within a FOV** — `load_data` is a plain `for` loop
  over loaders (`_tile_region.py:140-147`), while the dask graph runs them
  concurrently. `BY_TILE_DASK` *is* whole-image resident, but that is by design
  and the code says so: its docstring reads "Write tiles **in memory** … using
  Dask" and it logs "Starting Dask in-memory writing."
- **One block means one complete `__setitem__`, and the FOV loop is
  sequential**, so there is no per-block amplification and no race here at all.
  The corruption and the 84× read amplification documented above do not reach
  this file.

So the recommendation is **not** to chunk the lazy array at the write unit.
Doing that would split the single call into many, and on a region that is not
unit-aligned it would *introduce* a race that does not exist today. What is
actually available here:

1. Make `_compute_chunk_size` (`pipelines/_write_ome_zarr.py:33`) snap the
   target grid to the FOV grid, so FOV ROIs land unit-aligned. Today an
   unaligned FOV write read-modify-writes its edge units, and with sharding —
   which this path does not currently request — a FOV smaller than a shard
   would RMW the whole shard once per FOV, several FOVs deep on the same shard.
   Alignment is a caller-side property no writer can supply.
2. **Only then** parallelise across FOVs: `dask_parallel_fov_writing` is already
   a loop, so it is one `ThreadPoolExecutor` away from the `eager/threads`
   shape — safe precisely because (1) makes the ROIs unit-aligned and disjoint.
   Without (1), parallelising that loop is the bug this report documents.

This ordering is the whole recommendation. Step 2 without step 1 is a
regression, and step 2 is where the speedup is.

### `fractal-tasks-core/.../_projection_utils.py`

Replace `proj_image.set_array(dest_dask)` with the output-unit-driven reduce:
**322 ms against 2667 ms**, bounded memory, zero reads, no lock — which matters
here specifically, because Fractal's parallelism is processes and the lock ngio
relies on is not. The existing dtype workarounds (`safe_sum` clipping,
`mean_wrapper` recasting) carry over unchanged; they are about dask's type
promotion, not about the write.

Do not wait for `ngio.iterators` — as measured above, it cannot express this op
today.

### `ngio/common/_zoom.py` and `_pyramid.py`

1. **One-line fix, do it first:** `target.chunks` → `target.shards or
   target.chunks` at `_pyramid.py:47` and `:106`. Takes the sharded consolidate
   from 1024 reads / 1009 MiB to zero.
2. Then the halo-based unit-driven zoom as the lock-free replacement (846 ms
   against 3411 ms sharded) — **but gate it on unit size**, since it is 3.5×
   slower than numpy on a finely chunked unsharded target.

### ngio core

- **`set_slice_as_dask`** (`io_pipes/_ops_slices.py`): rechunk to the write unit
  as `ngio-dask-sharded-writes.md` proposes — and, if the lock is to go, take
  the `split` decomposition rather than the rechunk alone. The rechunk alone,
  lock-free, corrupts unaligned region writes (§1).
- **Fix the comment** at `_ops_slices.py:257-262`; it contradicts
  `_locks.py:8-10`, and the call-site version is the misleading one.
- **`ThreadPoolMapper`**: `MapperProtocol` already exists
  (`iterators/_mappers.py:17`) and `BasicMapper` is strictly sequential. A
  parallel mapper is ~15 lines at an extension point already designed for it,
  and `require_no_chunks_overlap()` is already there to gate it.
- **`by_write_unit()`**, or make `by_chunks()` follow `shards or chunks`. 4096
  ROIs for 16 write units is the bug in ROI form.
- **Output-driven iteration**, so a shape-changing op (reduce, resample) can be
  expressed at all — currently it cannot.
- **Read-halo / write-core ROIs**, so `overlap_xy` stops producing
  write-overlapping ROIs.
- If the lock survives at all, make it **per-array rather than module-global**:
  today two unrelated images serialise against each other.

### Backends

`tensorstore` is 1.6–3.7× faster than the best native strategy on every arm,
correct in all of them, and is the right answer if the dependency is
acceptable — with the caveat that it bypasses zarr-python entirely, so anything
ngio layers on its own store goes with it. `zarrs` was installed and
available but is **not** measured here — sizing it needs the `NgioStore`
`WrapperStore` blocker resolved first, which is
[`ngio-zarrs-codec-pipeline.md`](ngio-zarrs-codec-pipeline.md)'s subject rather
than this one's. The harness takes `--pipeline zarrs` when someone wants that
number.

---

## 7. Scope of the claim

zarr-python 3.3.0, dask 2026.7.1, numpy 2.5.2, ngio 1.0.0, tensorstore 0.1.85,
cubed 0.28.0, zarrs 0.2.3, scipy 1.18.0, Python 3.13.9, macOS 26.6.1 arm64.
`LocalStore` only, dask's threaded scheduler, single process, 8 workers, median
of 3 trials. Not tested against object stores or a distributed scheduler.

**Measured**: every strategy in §4 and §5, on both target layouts.
**Not measured**: `zarrs` (installed, but blocked on the `WrapperStore` issue);
multi-*process* writing — the multi-process claims here are arguments from the
read counts, not observations, and a zero read count is what makes them
arguments rather than guesses.

**Read `wrong` as worst-of-trials, not median.** One corrupt trial proves lost
updates; a clean one proves nothing. Arms whose reads cannot be counted —
tensorstore, which writes from C++ past the instrumented store — report `n/a`
rather than `0`.

**One arm is reported and should not be trusted as a recommendation**:
`eager/batched` is slower here than `eager/threads` because the harness's source
patch is a materialized array with no shared loads to deduplicate. Its point is
the converter's real graph, which this harness does not reproduce.

## Reproducing

```bash
uv run python reports/lazy_write_strategies.py --quick            # ~3 min
uv run --with tensorstore --with zarrs --with cubed \
       python reports/lazy_write_strategies.py                    # the numbers above
uv run python reports/lazy_write_strategies.py --emit-snippets    # refresh §4
```
