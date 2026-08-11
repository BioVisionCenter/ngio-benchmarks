# `da.store` into a sharded zarr array silently writes the wrong data

**Status:** confirmed, with a committed harness — `reports/dask_store_lock_rmw.py`
**Affects:** zarr-python 3.3.0 with dask 2026.7.1, any `da.store` whose blocks are smaller than the target's write unit
**Cost:** ~87% of a 128 MB array wrong, at full speed, with no error

## Summary

Writing a dask array into a zarr array with `da.store(..., lock=False)` is safe
when each dask block covers a whole **write unit**, and silently corrupting when
it does not. The write unit is the chunk — or the **shard**, when the array is
sharded.

That distinction is the whole finding, and it is easy to state wrongly. The
tempting phrasing is "a block that only *partially covers a chunk*". On a sharded
target that is inadequate: a block can cover its chunk exactly and still be 1/256
of the shard that actually gets written. So a **full-array write**, the case that
looks least suspicious, is the one that corrupts.

| write | dask block vs write unit | store reads | outcome unlocked |
| --- | --- | --- | --- |
| unsharded, full | 1 block = 1 chunk | **0** | correct |
| unsharded, chunk-aligned region | 1 block = 1 chunk | **0** | correct |
| unsharded, straddling region | block spans 4 chunks | 1024 | **corrupt** (contention-dependent) |
| **sharded, full** | **256 blocks per shard** | **4096** | **corrupt** |
| sharded, full, blocks chunked at the shard | 1 block = 1 shard | **0** | correct |
| sharded, full, plain `arr[...] = data` | one call, whole array | **0** | correct |

The three-line version: **zero store reads means no read-modify-write, and with
no read there is nothing to lose.** Where the reads are non-zero, concurrent
blocks read the same key, merge into their own stale snapshot, and write back
over each other.

The two halves of that are not equally strong, and the difference matters. A zero
read count is a **guarantee** — no read, no lost update, on any machine, at any
worker count. A non-zero read count is only an **exposure**: whether it actually
loses data depends on contention. The sharded full write below corrupted in every
trial at every worker count above one, and the straddling write corrupts at
128 MB — but the same straddling write on an 8 MB image, with 16 blocks instead
of 1024, came through clean in this repo's smoke run. Read the table as "safe" vs
"unsafe", never as "safe" vs "always visibly broken": the cases that pass by luck
today are the ones that fail on a bigger machine.

## Impact

A 128 MB `uint16` array, shape `(1, 64, 1024, 1024)`. The unsharded case is
`chunks=(1,1,256,256)`; the sharded case is `chunks=(1,1,128,128)` inside
`shards=(1,4,1024,1024)` — 256 chunks per shard, 16 shards. Median of 3.

| target | lock | workers | wall | cpu/wall | store reads | MiB read | wrong elements |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unsharded | `False` | 1 | 1747 ms | 0.86× | 0 | 0 | 0 |
| unsharded | `False` | 8 | **533 ms** | 2.86× | 0 | 0 | **0** |
| unsharded | `SerializableLock` | 8 | 1719 ms | 0.88× | 0 | 0 | 0 |
| sharded | `False` | 1 | 11081 ms | 0.82× | 4096 | 10816 | **0** |
| sharded | `False` | 2 | 4929 ms | 1.29× | 4096 | 5439 | **33,259,520** (49.6%) |
| sharded | `False` | 4 | 3459 ms | 1.64× | 4096 | 2713 | **50,135,040** (74.7%) |
| sharded | `False` | 8 | **3304 ms** | 1.76× | 4096 | 1405 | **58,277,888** (86.8%) |
| sharded | `SerializableLock` | 8 | 12305 ms | 0.83× | 4096 | 10816 | 0 |

Three things to read off this table.

**The corruption is not a rare interleaving.** Every trial at every worker count
above 1 was corrupt, and the fraction rises with parallelism: half the array at 2
workers, seven eighths at 8. The window is long — decode 256 inner chunks, merge,
re-encode, write ~2.4 MiB — and 4096 blocks contend over 16 keys, so the
*uncorrupted* outcome is the improbable one.

**`workers=1` is the control that separates the two claims.** It does the full
4096 read-modify-writes and loses nothing. Read-modify-write *happening* and
read-modify-write *racing* are different facts, and only the second needs a lock.

**Bytes read is the tell, and it moves the wrong way.** The lock does not reduce
the amplification: locked and serial-unlocked both read **10816 MiB to write 128
MiB**, an 84× read amplification either way. The racing runs read *less* — 1405
MiB at 8 workers — precisely because they are reading stale, half-written shards.
A drop in bytes read is the corruption's signature, not an optimisation.

For scale, the same write through this repo's `compare-io` suite, `sharded`,
`write_full`:

| row | median | store on disk | checksum |
| --- | --- | --- | --- |
| `ngio` `mode=numpy` | 593.7 ms | 84.5 MiB | `11a2b2a151f0` |
| `ngio` `mode=dask` | **13558.9 ms** | 84.5 MiB | `11a2b2a151f0` |
| `dask` | 3103.6 ms | **16.8 MiB** | **`38be1d40f420`** |
| `zarr` | 674.6 ms | 84.5 MiB | `11a2b2a151f0` |
| `tensorstore` | 27.8 ms | 128.0 MiB | `11a2b2a151f0` |

The `dask` row is 4.6× faster than ngio's dask path and wrote a store one fifth
the size, because most of its blocks never survived. Before the `checksum` column
covered writes, that row was simply "dask is fast here".

## Mechanism

Five steps, each verified against the installed zarr-python 3.3.0.

1. **`da.store` issues one `__setitem__` per dask block.** `load_store_chunk`
   (`dask/array/core.py:4762`) ends in `out[index] = x`, the ordinary
   `zarr.Array.__setitem__`. 4096 blocks means 4096 independent writes, each with
   its own indexer.

2. **A write that covers a whole chunk does not read.** The codec pipeline passes
   `None if is_complete_chunk else byte_setter` into `_read_key`
   (`zarr/core/codec_pipeline.py:414-431`), which short-circuits on `None`.
   `is_complete_chunk` is computed per dimension at `zarr/core/indexing.py:465-467`
   (`start == 0 and stop >= dim_limit and step in [1, None]`) and ANDed across
   dimensions at `:620-621`.

3. **On a sharded array the unit is the shard, and the fast path almost never
   applies.** `ShardingCodec._encode_partial_single`
   (`zarr/codecs/sharding.py:1350-1406`) takes the fast path only when
   `_is_complete_shard_write` (`:1437-1445`) holds — which requires the write to
   touch **every** inner chunk of the shard *and* every projection to be
   complete. One dask block touches 1 of 256. Otherwise it calls
   `_load_full_shard_maybe` (`:1596-1605`), which is
   `byte_getter.get(prototype=...)` with **no byte range**: the entire shard file
   is pulled back, merged, and rewritten.

4. **Both shipped pipelines do this, through different methods.** The default is
   `BatchedCodecPipeline` (`zarr/core/config.py:108-115`), which reads via the
   async `get`; `FusedCodecPipeline` reads via `get_sync`
   (`sharding.py:774-830`, `codec_pipeline.py:1227-1230`). The harness counts
   both and they agree — 0 reads unsharded, 4096 sharded, on each.

5. **Concurrent read-modify-writes on one key lose updates.** Nothing in zarr
   serialises them; `lock=` is dask's, and `False` is dask's default. Two blocks
   that load the same shard both merge into their own snapshot and the second
   write wins entirely.

## Why this is easy to miss

Nothing reports it. The write returns cleanly, no warning is emitted, and the
store is a valid zarr array afterwards — just one holding the wrong pixels. The
only visible symptom without an explicit check is that the store is *smaller*
than it should be, and "smaller" reads as "compressed better".

The predicate is also genuinely slippery, and this repo got it wrong in three
places before measuring it. ngio states it correctly where the lock is defined —
`ngio/common/_locks.py:8-10`, "does not align with the target's write unit — the
chunk, or the shard when the array is sharded" — and then loses it at the call
site, `ngio/io_pipes/_ops_slices.py:257-262`, which says "a **region** write
whose blocks only partially cover a chunk (or, for a sharded target, a shard)".
The "region write" qualifier is the error: on a sharded target a *full* write is
affected identically, and it is the more common call.

## Suggested fixes

**Chunk the patch at the write unit.** The harness arm `blocks=shard` takes the
same write from 4096 reads and 10816 MiB to **0 and 0**, with no lock and no
corruption, in 16 blocks instead of 4096. This is the actual fix: it removes the
amplification, which the lock does not.

**Failing that, keep the lock — but know what it costs.** On an unsharded
`write_full` it buys nothing at all (there are no reads to race) and costs 3.2×:
533 ms unlocked at 8 workers against 1719 ms locked. On a sharded target it is
the only thing making the output correct, but it leaves the 84× read
amplification untouched.

**Upstream, ngio:** rechunking the patch to the write unit inside
`set_slice_as_dask` takes ngio's own sharded `set_array` from **10697 ms to
581 ms**, byte-identical, against 442 ms for its numpy path — measured, not
projected. The call-site comment at `_ops_slices.py:257-262` also needs
correcting to match `_locks.py:8-10`. Both are written up for filing upstream in
[`ngio-dask-sharded-writes.md`](ngio-dask-sharded-writes.md), which is
self-contained and does not depend on this repo.

**Note on scope.** `SerializableLock` is per process. Under a distributed
scheduler it would not prevent cross-process lost updates, so the correct write
there is the rechunking, not the lock.

## Reproducing

```bash
uv run python reports/dask_store_lock_rmw.py --quick    # ~1 min
uv run python reports/dask_store_lock_rmw.py            # the numbers above
```

The harness counts store reads by subclassing `LocalStore` and overriding `get`,
`get_sync` and `get_partial_values`. Two details it depends on: the tally is
snapshotted **before** the verification readback (reading the array back issues
reads of its own, which produced a false positive the first time this was
written), and the geometry — blocks, write units, blocks per unit — is computed
and printed per arm rather than assumed, so the claim is scoped by arithmetic
that travels with the image specs.

What the arms prove, and what they do not: a positive result is conclusive — one
corrupt trial proves lost updates. A negative on the unsharded arm proves nothing
*on its own*, because absence of an observed race never does. What upgrades it to
a proof is the read counter reading zero. The mechanism is the proof; the trials
only confirm the instrument agrees.

## Environment

zarr-python 3.3.0, dask 2026.7.1, numpy 2.5.2, ngio 1.0.0, Python 3.13, macOS
(darwin 25.6.0), `LocalStore`, dask threaded scheduler, single process. Not
tested against object stores, `zarrs`, or a distributed scheduler.
