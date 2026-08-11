"""The comparison suites: ngio against the other libraries that do the job.

Two of them, because two different questions are being asked.

`compare-io` is about *array access*: read and write a full array, a
chunk-aligned region, and a region that straddles chunk boundaries. Its peers
are chunked-array engines -- zarr-python with and without the zarrs codec
pipeline, dask, tensorstore, z5py -- so the question is what ngio's layer costs
over the metal, and where that layer earns it back.

`compare-create` is about *building an OME-Zarr*: writing a multiscale pyramid
from an array in memory. Its peers are the other OME-Zarr writers -- ngff-zarr,
ome-zarr-py, iohub, bioio, acquire-zarr.

Every implementation runs in its own isolated environment. See `core.envs` for
why that is unconditional rather than a fallback.

## The adapter protocol

An adapter is a module that declares some constants and one function:

    NAME     = "tensorstore"
    REQUIRES = ("tensorstore>=0.1.85", "numpy>=2")   # its uv environment
    SUPPORTS = frozenset({"read_full", ...})         # operations it can express
    FORMATS  = frozenset({2, 3})                     # zarr formats it handles
    PYTHON   = None                                  # interpreter pin, if any
    METHODS  = {"mean": ..., "nearest": ...}         # canonical name -> its own
    OPTIONS  = {"mode": ["dask", "numpy"]}           # its own settings, as axes

    def build(op, spec, root, *, method=..., **options) -> Measured: ...

The constants are read by the **parent**, which does not have any of these
libraries installed. So an adapter must be importable with nothing but the
standard library and numpy: every `import tensorstore` belongs inside `build`,
never at module scope. This is the same discipline the `algorithms` block
follows, for the same reason -- a module that cannot be imported must degrade
one column, not the run.

`SUPPORTS` and `FORMATS` are declarations, not documentation. A case they
exclude is reported as `unsupported` in the table rather than left blank,
because in a comparison an unexplained gap reads as a result. `METHODS` is the
same kind of declaration for the downsampling filter: a canonical name it does
not map is `unsupported` for that row rather than quietly served by the
library's default.

`METHODS` and `OPTIONS` are both optional. An adapter that declares neither --
every adapter in `compare-io` does -- is called as `build(op, spec, root)`: the
parent only passes keywords an adapter declared, so there is no `**options` to
absorb on the modules that have none.

`OPTIONS` follows `core.axes`' two declaration forms exactly. A list is an
**open** axis and a config may name values it does not list; a mapping is
**closed** and a config may only subset its labels. Choose by whether the
wrapped library validates the value itself -- if it does, declaring a closed
axis here only means going stale when the library grows an option.
"""
