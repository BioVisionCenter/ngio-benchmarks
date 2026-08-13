"""ngio, through its public image API.

Reads go through `open_image` and the slicing keywords rather than the `Roi`
objects the internal `roi` block uses. Two reasons: `ngio.common.Roi` is not
part of the public surface an adapter should depend on, and a keyword slice is
the same *kind* of request every other library in this suite receives, so the
comparison is of the machinery underneath rather than of two different ways of
asking.

The container ngio writes into is created during setup and excluded from the
timing, which is the right split -- the peers here are also handed an existing
array to write into.

`mode` picks which of ngio's two array types the operation goes through:
`get_as_numpy` and a numpy patch, or `get_as_dask` and a dask one. On the read,
`.compute()` is **inside** the timed callable, and so is `get_as_dask` itself --
`da.from_zarr` is called inside it and is not separable from it, and the graph
ngio builds per call is exactly the layer cost this suite prices. A row that
measured only graph construction would be the fastest in the table and would
mean nothing. The write needs no `.compute()`: `set_array` dispatches on the
patch's type and its dask branch ends in an eager `da.store`.

One thing to know before reading `mode=dask` against the `dask` column beside
it: **they do not always take the same path to correctness.** Released ngio
(1.0.0) stores through `da.store(..., lock=DASK_STORE_LOCK)`, a process-global
lock shared by every dask write in the library, so its flushes are serialised
regardless of whether the block about to flush actually needs it. The `dask`
column stores through `da.to_zarr`, which asks the target for its write
unit -- the chunk, or the *shard* when the array is sharded -- and rechunks
the patch to a multiple of it before storing, so none of its blocks can
straddle a unit two writers might touch at once and no lock is needed.

What ngio's lock buys is not the same in every cell. zarr read-modify-writes a
whole write unit whenever a block does not cover one, and two blocks racing on
the same unit lose an update. On a `write_full` into an unsharded spec each
dask block *is* exactly one chunk, so zarr takes its complete-chunk path and
issues no store read at all: nothing to race, and the lock costs ~3x for
nothing. On `sharded` the unit is a 256-chunk shard that no single block can
fill, so even `write_full` is 4096 read-modify-writes over 16 keys -- and an
unlocked `da.store(..., lock=False)` really does write corrupt data there,
~87% of the array wrong, as measured in
`reports/dask-sharded-write-races.md`. The straddling ops are that second case
on any layout. So on `sharded` and on either straddling write, the gap to the
`dask` column is close to the cost of buying that same correctness the slow
way rather than by rechunking first; on an unsharded `write_full` it is close
to pure overhead. `checksum` covers writes as well as reads so neither claim
has to be taken on trust.

The reads carry no such asymmetry: both build a graph over the same chunk grid,
because `da.from_zarr` takes its chunks from `Array.chunks`, which is the *inner*
chunk shape and so is `spec.chunks` on a sharded image too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ngio_benchmarks.compare.io import _ops
from ngio_benchmarks.compare.io._fixture import store_path
from ngio_benchmarks.core.measure import Measured

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

NAME = "ngio"
DISTRIBUTION = "ngio"
REQUIRES = ("ngio",)
SUPPORTS = frozenset(_ops.READS + _ops.WRITES)
FORMATS = frozenset({2, 3})
PYTHON = None

#: Which of ngio's two array types the operation goes through.
#:
#: Closed, unlike the same-named option on the create adapter. There `mode` is a
#: string `Image.consolidate` validates itself, so the axis stays open and does
#: not go stale when ngio grows a fourth value; here there is no `mode=`
#: parameter anywhere in ngio -- this module is choosing between two methods, so
#: this table *is* the vocabulary. Same reason `bioio`'s `writer` is closed.
#: It also means a typo is a `--list` error rather than a failed row discovered
#: after six environments installed.
OPTIONS = {"mode": {"numpy": "numpy", "dask": "dask"}}


def _slicing(spec: ImageSpec, op: str) -> dict[str, slice]:
    """The trailing-axis slices as ngio's keyword arguments.

    Empty for a full read: ngio takes the whole array when nothing is named,
    and passing explicit full slices would measure a slicing path the other
    libraries are not being asked to take.
    """
    if _ops.is_full(op):
        return {}
    index = _ops.region(spec, op)
    return {
        name: index[i] for i, name in enumerate(spec.axes) if i >= len(spec.shape) - 2
    }


def build(op: str, spec: ImageSpec, root: Path, **options: str) -> Measured:
    """The measured callable for one operation."""
    from ngio import open_image

    # Defaulted here rather than by the runner, which passes only the options a
    # config actually named -- so a config that never mentions `mode` measures
    # exactly what this adapter measured before the axis existed.
    mode = options.get("mode", "numpy")
    slicing = _slicing(spec, op)

    if _ops.is_read(op):
        image = open_image(store_path(root, spec), path="0", mode="r")
        if mode == "dask":
            return Measured(lambda: image.get_as_dask(**slicing).compute())
        return Measured(lambda: image.get_as_numpy(**slicing))

    path = _ops.target(root, NAME, op, spec)
    image = _create(path, spec)
    data = _ops.patch(spec, op, root)
    if mode == "dask":
        import dask.array as da

        # Out here, and chunked exactly as the `dask` column chunks its own
        # patch: the two write rows should differ in ngio's layer, not in the
        # array they were handed. Wrapping is also the same fold `_ops.patch`
        # already refuses -- a patch built lazily from the source would put that
        # read inside the timing of the write.
        data = da.from_array(data, chunks=spec.chunks)
    return Measured(
        lambda: image.set_array(patch=data, **slicing), extra={"target": str(path)}
    )


def _create(path: Path, spec: ImageSpec):
    """An empty single-level container for the write operations."""
    from ngio import create_empty_ome_zarr

    container = create_empty_ome_zarr(
        store=path,
        shape=spec.shape,
        axes_names=list(spec.axes),
        channels_meta=[f"Channel {i + 1}" for i in range(spec.shape[0])]
        if spec.axes[0] == "c"
        else None,
        levels=1,
        pixelsize=spec.pixelsize,
        chunks=spec.chunks,
        shards=spec.shards,
        ngff_version=spec.ngff_version,
        overwrite=True,
    )
    return container.get_image(path="0")
