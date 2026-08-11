# ngio's dask write path is 18× slower than it needs to be on sharded arrays

**Status:** confirmed, with a standalone reproduction and a verified fix
**Affects:** ngio 1.0.0 with zarr-python 3.3.0 — `set_array` / `set_roi` / the iterators, any dask patch into a **sharded** store
**Cost:** 10.7 s where 0.58 s suffices, and 84× read amplification, on every sharded dask write
**Fix:** rechunk the patch to the store's write unit before `da.store` — three lines in `set_slice_as_dask`

## Summary

`ngio.io_pipes._ops_slices.set_slice_as_dask` hands the caller's dask array
straight to `da.store`, so the block grid is whatever the caller chose. When the
target is **sharded**, that is almost never the grid zarr writes in: zarr's write
unit is the *shard*, and a patch chunked at the array's `chunks` gives 256 blocks
per shard.

zarr then read-modify-writes the entire shard once per block. Not once per shard —
once per **block**. A full write of a 128 MB sharded array issues 4096 whole-shard
reads over 16 keys and moves 10.6 GB to write 128 MB.

`DASK_STORE_LOCK` makes this correct, and it is genuinely needed — without it the
same write loses ~87% of the array. But the lock treats the symptom. Rechunking
the patch to the shard shape removes the read-modify-write altogether: the reads
go to **zero**, the lock has almost nothing left to serialise, and the write drops
from 10.7 s to 0.58 s — within 30% of the numpy path.

| `set_array` on a sharded store, 128 MB | wall | correct | store |
| --- | --- | --- | --- |
| numpy patch | 442 ms | yes | 109.6 MiB |
| dask patch, blocks = `chunks` — **what ngio does today** | **10697 ms** | yes | 109.6 MiB |
| dask patch, blocks = `shards` — **the fix** | **581 ms** | yes | 109.6 MiB |

Byte-identical output, same store size, 18.4× faster.

## Reproduction

Standalone — ngio, numpy, dask, nothing else:

```python
import shutil, time
from pathlib import Path
import numpy as np, dask.array as da
from ngio import create_empty_ome_zarr

SHAPE, CHUNKS, SHARDS = (1, 64, 1024, 1024), (1, 1, 128, 128), (1, 4, 1024, 1024)
root = Path("/tmp/ngio-shard-write"); shutil.rmtree(root, ignore_errors=True)
src = np.random.default_rng(0).integers(1, 4000, size=SHAPE, dtype=np.uint16)

def image(name):
    container = create_empty_ome_zarr(
        store=root / name, shape=SHAPE, axes_names=list("czyx"), levels=1,
        pixelsize=1.0, chunks=CHUNKS, shards=SHARDS, ngff_version="0.5",
        overwrite=True,
    )
    return container.get_image(path="0")

for label, blocks in (("numpy", None), ("blocks=chunks", CHUNKS), ("blocks=shards", SHARDS)):
    img = image(label)
    patch = src if blocks is None else da.from_array(src, chunks=blocks)
    start = time.perf_counter()
    img.set_array(patch=patch)
    wall = time.perf_counter() - start
    print(f"{label:16} {wall * 1000:8.0f} ms   wrong={int((img.get_as_numpy() != src).sum()):,}")
```

```
numpy                 442 ms   wrong=0
blocks=chunks       10697 ms   wrong=0
blocks=shards         581 ms   wrong=0
```

## Mechanism

1. **`da.store` issues one `zarr.Array.__setitem__` per dask block.**
   `load_store_chunk` (`dask/array/core.py:4762`) ends in `out[index] = x`. A
   patch chunked at `(1,1,128,128)` over `(1,64,1024,1024)` is 4096 blocks.

2. **zarr skips the read only for a write that covers a whole unit.** The codec
   pipeline passes `None if is_complete_chunk else byte_setter` into `_read_key`
   (`zarr/core/codec_pipeline.py:414-431`), which short-circuits on `None`.

3. **On a sharded array the unit is the shard, and one block never fills it.**
   `ShardingCodec._encode_partial_single` (`zarr/codecs/sharding.py:1350-1406`)
   takes its fast path only when `_is_complete_shard_write` (`:1437-1445`) holds —
   every inner chunk touched, every projection complete. One block touches 1 of
   256. So each block falls into `_load_full_shard_maybe` (`:1596-1605`), which is
   `byte_getter.get(prototype=...)` with **no byte range**: the whole shard file,
   decoded, merged, re-encoded, rewritten.

4. **That is the 84× amplification**, and `DASK_STORE_LOCK` does not reduce it —
   it only stops the concurrent versions of it from overwriting each other.
   Measured on the identical geometry with a counting `LocalStore`:

   | patch blocks | store reads | keys | MiB read (to write 128 MB) |
   | --- | --- | --- | --- |
   | `chunks` (today) | 4096 | 16 | 10816 |
   | `shards` (fix) | **0** | **0** | **0** |

## Suggested fix

In `set_slice_as_dask` (`ngio/io_pipes/_ops_slices.py`), align the patch to the
target's write unit before storing. The unit is `zarr_array.shards or
zarr_array.chunks`; `da.rechunk` is a graph operation, so this costs nothing at
call time and strictly reduces the number of `__setitem__` calls:

```python
unit = zarr_array.shards or zarr_array.chunks
if patch.chunksize != unit:
    patch = patch.rechunk(unit)
da.store(patch, zarr_array, regions=slice_tuple, lock=DASK_STORE_LOCK)
```

Two caveats worth handling deliberately rather than by accident:

- **A rechunk to a shard-sized block raises peak memory** to one shard per
  in-flight task (4 MiB here). Worth gating on a size ceiling if that matters,
  but note it replaces 256 *decode-merge-re-encode* cycles per shard, so the
  allocation it adds is smaller than the one it removes.
- **A region write that does not start on a unit boundary still cannot be made
  complete**, so the lock must stay. The fix removes the read-modify-write in the
  aligned cases, which includes every full-array write; it does not remove the
  need for the lock in general.

## Also: the comment at the call site says the wrong thing

`ngio/common/_locks.py:8-10` states the rule correctly:

> A block whose footprint **does not align with the target's write unit** — the
> chunk, or the shard when the array is sharded — makes zarr read-modify-write
> that unit

`ngio/io_pipes/_ops_slices.py:257-262` then restates it as:

> The shared lock serialises the flushes: **a region write** whose blocks only
> partially cover a chunk (or, for a sharded target, a shard) makes zarr
> read-modify-write it

The "region write" qualifier is wrong, and it is the misleading half. On a sharded
target a **full** write is affected identically — it is in fact the worst case,
because it touches every shard — and it is the more common call. Anyone reading
the call-site comment concludes that `set_array` with no slicing kwargs is the
safe, fast path. It is the slow one.

(This is not a hypothetical misreading: three separate places in the downstream
benchmark repo that prompted this investigation had copied exactly that phrasing
and drawn exactly that conclusion.)

## Scope of the claim

zarr-python 3.3.0, ngio 1.0.0, dask 2026.7.1, `LocalStore`, dask's threaded
scheduler, single process, macOS. The read counts were verified on both codec
pipelines zarr ships (`BatchedCodecPipeline`, the default, reads through `get`;
`FusedCodecPipeline` through `get_sync`) and agree. Not tested against object
stores, `zarrs`, or a distributed scheduler — note in passing that
`SerializableLock` is per-process, so under a distributed client the lock would
not prevent cross-process lost updates and the rechunk becomes the only correct
answer rather than merely the faster one.

Full measurements, including the lost-update sweep that shows what the lock is
protecting against, are in `dask-sharded-write-races.md` alongside this file, with
the harness that produced them in `dask_store_lock_rmw.py`.
