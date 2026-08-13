# Chunk and shard layout for OME-Zarr v3: keep the FOV chunk

**Status:** measured, synthetically and on real plate data — `zarr-performance-exploration`, `pixi run layout`
**Question:** moving to zarr v3, should an FOV-chunked plate adopt sharding?
**Answer:** **no, for FOV-aligned and whole-image access** — sharding costs 2.3× on whole-image reads and buys nothing those patterns need
**But:** sharding cuts sub-FOV read amplification from **12.6× to 0.8×**, which matters if masked-ROI access is on your roadmap

---

## 1. Why zarrs looked weaker on real data than the benchmarks promised

Two candidate explanations. Both are real, and they apply to different halves of
the workload.

### Explanation A: the data is zarr v2, so zarrs never ran

`20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0/0` is zarr **v2**
— it has `.zarray`, not `zarr.json`. zarrs replaces the **v3** codec pipeline;
against a v2 array it declines, **silently**, and zarr-python falls back to
`BatchedCodecPipeline` with no warning.

A zarrs evaluation on that dataset measured zarr-python twice. This is the same
silent-fallback behaviour documented in
[`ngio-upstream-write-path.md`](ngio-upstream-write-path.md) §5, where a plain
`LocalStore` *subclass* is also refused.

**Check before believing any zarrs number:**

```python
type(array._async_array.codec_pipeline).__name__   # want: ZarrsCodecPipeline
```

### Explanation B: large chunks amortise away the overhead zarrs removes — for reads only

169 MB uint16, unsharded, chunk size varied 64×, median of 3:

| chunk | chunk MB | chunks | **read_full** speedup | **write_full** speedup |
| --- | --- | --- | --- | --- |
| 270×320 | 0.16 | 1024 | **6.10×** | 4.15× |
| 540×640 | 0.66 | 256 | 3.26× | 4.40× |
| 1080×1280 | 2.64 | 64 | 2.26× | 4.35× |
| **2160×2560** (your FOV) | **10.55** | 16 | **1.86×** | **3.57×** |

**Reads: the hypothesis is confirmed.** The zarrs win falls monotonically from
6.1× to 1.86× as chunks grow 64×. Per-chunk Python overhead is roughly constant,
so at 10.55 MB there is little left for a Rust pipeline to remove.

**Writes: the hypothesis does not hold.** The win is essentially flat, 4.15× →
3.57×. Compression dominates write cost at every chunk size, and zarrs' blosc is
faster regardless of how the data is divided.

So the two explanations split cleanly: **if you saw weak zarrs benefit on
*reads*, large chunks explain it. If you saw weak benefit on *writes*, chunk size
does not — the v2 fallback does.**

On real pixels at the real chunk size, layout A gives **1.52× on `read_full` and
3.94× on `write_full`**. That is the honest answer to "what would zarrs buy me".

---

## 2. Should an FOV-chunked plate adopt sharding?

Five layouts, all v3, uint16, blosc lz4/5/shuffle, 169 MB, FOV = 2160×2560:

| id | chunks | shards | codec unit | write unit | files |
| --- | --- | --- | --- | --- | --- |
| **A** | 2160×2560 | — | 10.55 MB | 10.55 MB | 17 |
| **B** | 540×640 | — | 0.66 MB | 0.66 MB | 257 |
| **C** | 540×640 | 2160×2560 | 0.66 MB | 10.55 MB | 17 |
| **D** | 270×320 | 2160×2560 | 0.16 MB | 10.55 MB | 17 |
| **E** | 2160×2560 | 4320×5120 | 10.55 MB | 42.19 MB | 5 |

### For the patterns that matter here, A wins

zarr-python column, which is what ships today:

| pattern | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| `read_full` | **38 ms** | 85 ms | 88 ms | 138 ms | 58 ms |
| `read_fov` | **4 ms** | 6 ms | 8 ms | 14 ms | 5 ms |
| `write_full` | **133 ms** | 221 ms | 156 ms | 230 ms | 144 ms |
| `write_fov` | 16 ms | **13 ms** | 15 ms | 22 ms | 16 ms |

A is fastest on three of the four, and within 3 ms of B on the fourth.

Sharding costs **2.3× on whole-image reads** (38 → 88 ms) and **2× on FOV reads**
(4 → 8 ms), because the same bytes now pass through 16× more codec calls. Real
pixels agree: `read_full` A 42 ms against C 98 ms.

Under zarrs the gap narrows (A 21 ms, C 23 ms on `read_full`) — the Rust pipeline
absorbs the extra codec calls. But A is never worse, and zarrs is not available
today.

**FOV writes are clean under both A and C** — zero store reads, no
read-modify-write — because in both, one FOV is exactly one write unit. That is
the property worth protecting, and it is what layout E breaks.

### Sharding's one decisive win, which you deselected

| pattern | A read amp | C read amp |
| --- | --- | --- |
| `read_sub_fov` (512², synthetic) | **21.1×** | **1.3×** |
| `read_sub_fov` (512², real pixels) | **12.6×** | **0.8×** |

A 512² crop out of layout A must decompress a whole 10.55 MB chunk. Under C it
touches one 0.66 MB inner chunk. That is a **16× reduction in bytes moved**, and
it is the entire case for sharding.

This was outside the scoped patterns, and it is measured anyway because
**Fractal's masked ROI tables read per-object regions much smaller than an FOV**.
If `MaskedImage` workloads matter, this row is the one to weigh; if they do not,
ignore it.

C also gets B's codec granularity at A's file count — 17 files, not 257 —
which is the other classic sharding argument, also deselected here.

### Layout E is the trap

E sets the shard larger than the FOV — a natural-seeming "bigger is better"
choice. Measured on `write_fov`: **1 store read where A and C have 0.** The FOV
write is now a *partial* shard write, so zarr read-modify-writes the whole
42 MB shard. At this scale it is cheap; at plate scale, with concurrent writers,
it is the failure mode documented in
[`dask-sharded-write-races.md`](dask-sharded-write-races.md).

**If you shard, the shard must be exactly the FOV.**

### Compression does not enter into it

All five layouts produced a store within 0.1 MiB of each other (168.8 MiB
synthetic; 94.1 vs 94.0 MiB on real pixels). Chunk size does **not** trade
compression ratio against access cost here, so that consideration can be dropped
from the decision.

---

## 3. What this means for the stack

### Keep chunking at the FOV

The current convention — `_compute_chunk_size` deriving xy chunks from the first
FOV's shape — is **the right one and should survive the move to v3 unchanged.**
It makes every FOV read exactly one chunk and every FOV write exactly one
complete write unit. Nothing measured here improves on that for FOV-aligned or
whole-image access.

This is a "no change needed" result, and it is worth stating plainly: the v3
migration does not require a layout redesign.

### Adopt sharding only if sub-FOV access is on the roadmap

Adopt C — **chunks 540×640 inside shards of exactly one FOV** — if and only if
one of these applies:

- **Masked-ROI / per-object access matters** (12.6× → 0.8× bytes read), or
- **Viewport / interactive visualisation matters** (same mechanism), or
- **File count matters** operationally — object-store listing, rsync, inodes.

The cost is 2.3× on whole-image reads under zarr-python today, shrinking to
~1.1× if zarrs is ever adopted. Sharding and zarrs are complements: sharding
adds codec calls, and zarrs is what makes them cheap.

### Where the defaults live

Two places, if any of this is acted on:

- `ome-zarr-converters-tools/src/ome_zarr_converters_tools/pipelines/_write_ome_zarr.py:33`
  — `_compute_chunk_size` derives xy chunks from the first FOV and never sets
  `shards`. To adopt C, it would set `chunks = FOV / 4` and `shards = FOV`.
- ngio's `create_empty_ome_zarr(chunks="auto", shards=None)` — the `shards=None`
  default is the right one on this evidence.

### The zarrs decision depends on the layout you ship

Whether to pursue the `WrapperStore` unwrap
([`ngio-upstream-write-path.md`](ngio-upstream-write-path.md) §5) is not a pure
code question. At FOV-sized chunks zarrs is worth **1.5× on reads and 3.9× on
writes**; at 0.16 MB chunks it is worth 6×. Decide against the layout you
actually intend to ship — and note that the write win is the chunk-size-robust
one.

---

## 4. Scope of the claim

zarr-python 3.3.0, zarrs 0.2.3, numpy 2.5.2, Python 3.13.9, macOS 26.6.1 arm64,
`LocalStore`, single machine, median of 3. 169 MB arrays at real FOV geometry;
real-pixel confirmation is channel 0, a 4×4-FOV corner of
`.../B/03/0/0`, at identical geometry to the synthetic run.

**Measured:** every table. zarrs engagement is verified per cell — a fallback
would be labelled, not silently reported as 1.0×.

**Qualified.** `read amp` is MiB read (compressed, off disk) over MiB requested
(uncompressed), so values below 1.0 reflect the compression ratio rather than an
absence of amplification; the column is meaningful *across layouts for one
pattern*, which is how it is used. Sub-FOV wall times are near the noise floor at
this array size — the amplification figure is the signal there, not the clock.

**Not measured:** object stores, where per-request latency would amplify the
file-count and read-amplification differences considerably; multi-channel and
multi-z chunking (every layout here chunks one channel-plane at a time, as the
real data does); pyramid levels beyond level 0.

## Reproducing

```bash
cd zarr-performance-exploration
pixi run layout --trials 3
pixi run layout --trials 3 --real /path/to/plate.zarr/B/03/0/0
```
