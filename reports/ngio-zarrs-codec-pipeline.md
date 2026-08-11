# ngio cannot use the zarrs codec pipeline

**Status:** confirmed, with a one-line reproduction
**Affects:** ngio 1.0.0 with zarr-python 3.3.0 and zarrs 0.2.3
**Cost:** ~1.6–1.7× on pyramid creation, on every store, silently

## Summary

`zarrs` accelerates zarr-python by replacing its codec pipeline process-wide:

```python
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```

Every OME-Zarr writer measured in this repo picks that up and gets faster.
**ngio does not.** It gets nothing — not a small gain, exactly nothing — because
it wraps every store in `NgioStore`, and zarrs refuses to install its pipeline
for a wrapped store. zarr-python then falls back to its own pipeline **without a
warning**, so the configuration appears to have been accepted.

The wrapper that costs ngio this is, with ngio's default configuration, a pure
pass-through: `RetryConfig.max_retries` defaults to `0`, and `NgioStore`'s own
docstring says that with that policy "every IO call is delegated as-is". So by
default the wrapper's only observable effect is to disable zarrs.

## Impact

Measured directly: ngio building a 4-level pyramid from a 128 MB `uint16` array
(shape `(1, 64, 1024, 1024)`, chunks `(1, 1, 256, 256)`, zstd), median of 3 runs
after a warmup. The third column shims in the unwrap that a fix would perform,
and is what ngio is leaving on the table.

| `consolidate` mode | zarr-python | zarrs, as ngio ships | zarrs, store unwrapped |
| --- | --- | --- | --- |
| `dask` | 1394.5 ms | 1431.4 ms — **0.97×** | 876.6 ms — **1.59×** |
| `numpy` | 936.2 ms | 934.7 ms — **1.00×** | 541.8 ms — **1.73×** |

For scale, the same 128 MB written by the other libraries in this repo's
`compare-create` suite, and by zarr-python directly:

| writer | zarr-python | zarrs | speedup |
| --- | --- | --- | --- |
| raw `arr[:] = data` (`compare-io`) | 315.7 ms | 111.4 ms | 2.83× |
| `ngff-zarr` | 4228.6 ms | 1121.4 ms | 3.77× |
| `bioio` | 2220.8 ms | 1218.6 ms | 1.82× |
| `ome-zarr-py` | 3435.0 ms | 3119.3 ms | 1.10× |
| **`ngio`** | **1394.5 ms** | **1431.4 ms** | **none** |

(`ome-zarr-py` is low for an unrelated and legitimate reason: 83% of its wall
clock is skimage's anti-aliased `resize`, which no codec pipeline touches. Its
`levels=1` write does speed up, 1.59×. ngio's does not speed up at all.)

## Mechanism

Five steps, each verified:

1. ngio normalises every store through `NgioStore.from_any` / `NgioStore.ensure`
   (`ngio/utils/_store.py:61`), a subclass of `zarr.storage.WrapperStore` that
   carries the retry policy.

2. When zarr-python builds an array's codec pipeline it calls
   `create_codec_pipeline` (`zarr/core/array.py:224`), which first tries the
   store-aware constructor:

   ```python
   if store is not None:
       try:
           return get_pipeline_class().from_array_metadata_and_store(
               array_metadata=metadata, store=store
           )
       except NotImplementedError:
           pass
   ```

3. zarrs rejects the wrapper outright:

   ```
   NotImplementedError: zarrs-python does not support WrapperStore stores
   ```

4. zarr-python catches that `NotImplementedError` and falls through to
   `from_codecs`. That `except ... : pass` exists for pipelines that do not
   implement the store-aware constructor at all — it cannot distinguish "this
   pipeline class has no such method" from "this pipeline rejected this specific
   store", and treats both as a silent fallback.

5. `ZarrsCodecPipeline.from_codecs` is itself a hand-back
   (`zarrs/pipeline.py:120`):

   ```python
   @classmethod
   def from_codecs(cls, codecs):
       return BatchedCodecPipeline.from_codecs(codecs)
   ```

   so the array ends up holding a plain `BatchedCodecPipeline`.

No warning is emitted at any point. `zarr.config.get("codec_pipeline.path")`
still reports `zarrs.ZarrsCodecPipeline` afterwards, so nothing in the process
indicates the pipeline is not the one that was asked for.

### There is no caller-side workaround

The wrapping happens regardless of what the caller passes:

| passed to `create_empty_ome_zarr(store=...)` | store on the array | pipeline |
| --- | --- | --- |
| `Path` | `NgioStore` | `BatchedCodecPipeline` |
| `str` | `NgioStore` | `BatchedCodecPipeline` |
| `LocalStore` | `NgioStore` | `BatchedCodecPipeline` |

## Reproduction

```python
import zarr
from ngio import create_empty_ome_zarr

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

def pipeline_of(array):
    return type(array._async_array.codec_pipeline).__name__

raw = zarr.create_array(store="raw.zarr", shape=(1, 16, 512, 512),
                        dtype="uint16", chunks=(1, 1, 256, 256))
print("plain zarr :", pipeline_of(raw))          # ZarrsCodecPipeline

container = create_empty_ome_zarr(store="ngio.zarr", shape=(1, 16, 512, 512),
                                  axes_names=["c", "z", "y", "x"], levels=3,
                                  pixelsize=(0.5, 0.5), chunks=(1, 1, 256, 256),
                                  dtype="uint16", ngff_version="0.5")
print("via ngio   :", pipeline_of(container.get_image(path="0").zarr_array))
#                                                     ^ BatchedCodecPipeline
```

To see the rejection directly:

```python
from zarr.storage import LocalStore, WrapperStore
from zarrs.pipeline import ZarrsCodecPipeline

ZarrsCodecPipeline.from_array_metadata_and_store(
    array_metadata=raw._async_array.metadata,
    store=WrapperStore(LocalStore("raw.zarr")),
)
# NotImplementedError: zarrs-python does not support WrapperStore stores
```

## Why this is easy to miss

The failure is invisible to every check one would normally make.

- **The output is byte-identical.** Which pipeline encoded a chunk does not
  change what lands on disk, so store size, codec metadata, chunk grid and
  pyramid shape all match whether or not the swap took. This repo's audit
  columns — which exist precisely to catch a writer that quietly did something
  else — cannot see it.
- **The configuration reads back correct.** `zarr.config` still says
  `zarrs.ZarrsCodecPipeline`.
- **Nothing warns.** zarrs has a `UserWarning` path for unsupported *metadata*,
  but the unsupported-store case raises `NotImplementedError` and is swallowed
  upstream before any warning is reached.

The one reliable tell is the measurement itself: **wall-clock and CPU seconds
both unchanged** means the same code ran. In the table above, ngio's CPU time
moved from 2.40 s to 2.46 s — noise — while every library that really switched
also changed how many cores it used.

The programmatic check is a single expression:

```python
type(image.zarr_array._async_array.codec_pipeline).__name__ == "ZarrsCodecPipeline"
```

## Suggested fixes

**In ngio, and this one is sufficient on its own:** do not wrap when the retry
policy is a no-op. `NgioStore.ensure` and `NgioStore.from_any` could return the
store unchanged when `max_retries == 0`, which is the default. That is no
behaviour change for anyone — the wrapper already delegates every call as-is
under that policy — and it restores zarrs for the default configuration. Users
who enable retries would keep the wrapper and forgo zarrs, which is at least a
trade they chose.

A fuller version would keep the wrapper and let it advertise the store beneath
it, so that a consumer needing a concrete store type can reach it. That needs
agreement with zarrs on the protocol, so it is the slower path.

**In zarrs-python:** consider unwrapping a `WrapperStore` whose subclass
declares itself a pass-through. Unwrapping unconditionally would be wrong — a
wrapper may legitimately transform keys or bytes — so this wants an opt-in
marker rather than an `isinstance` check.

**In zarr-python:** `create_codec_pipeline` swallowing `NotImplementedError`
means a user who explicitly configured a codec pipeline can silently get a
different one, with no diagnostic anywhere. Warning on that path would have made
this a five-minute discovery instead of a benchmark anomaly. This is arguably
the root reporting bug, independent of ngio and zarrs.

## Environment

| | |
| --- | --- |
| ngio | 1.0.0 |
| zarr-python | 3.3.0 |
| zarrs | 0.2.3 |
| numpy | ≥2 |
| Python | 3.13 |
| Platform | macOS (Darwin 25.6.0), arm64 |

Timings produced with this repo's `compare-create` suite plus a standalone
script for the unwrapped variant; the suite's `pipeline` axis is what surfaced
the anomaly. Every store was written to a local filesystem, which is a store
type zarrs fully supports — the wrapper is the only obstacle.
