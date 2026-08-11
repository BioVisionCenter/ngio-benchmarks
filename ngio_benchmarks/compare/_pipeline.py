"""Which codec pipeline zarr-python encodes and decodes through.

`zarrs` is not a library with an API of its own. It installs a replacement for
zarr-python's codec pipeline, so every library that writes through zarr-python
picks it up without knowing it exists -- which is what makes it an axis of both
comparison suites rather than a competitor in either.

The two suites take that differently, and both are right for their own shape.
In `compare-io` the measured code is one module either way, so `zarr` and
`zarrs` are two implementations whose delta is the pipeline and nothing else. In
`compare-create` there are five different libraries sitting above the pipeline,
so it is a `pipeline` axis every one of them is swept along. Either way the
vocabulary, the pin and the reasoning live here.

Set once, before the adapter builds, rather than around the timed callable:
`zarr.config.set` is a context the pipeline is chosen from at array-open time,
so toggling it inside the measurement would time the toggle.
"""

from __future__ import annotations

#: What zarr-python does untold. Named rather than left as an empty string, so a
#: `pipeline` column always says which stack encoded the row.
DEFAULT = "zarr-python"

#: Pipeline name -> the `codec_pipeline.path` that installs it. `None` is
#: zarr-python's own, which is installed by not touching anything.
PIPELINES: dict[str, str | None] = {
    DEFAULT: None,
    "zarrs": "zarrs.ZarrsCodecPipeline",
}

#: What a pipeline adds to whatever environment it runs in, on top of that
#: implementation's own `REQUIRES`. Added per case rather than per adapter so a
#: `zarr-python` row installs exactly the environment it installed before this
#: axis existed -- a baseline whose environment moved is not a baseline.
REQUIRES: dict[str, tuple[str, ...]] = {"zarrs": ("zarrs>=0.2.3",)}

#: Which zarr formats a pipeline can be installed for, where it is not both.
#: `zarrs` implements the zarr **v3** codec pipeline; v2 has no equivalent to
#: replace, so a v2 image under it would silently measure zarr-python twice.
FORMATS: dict[str, frozenset[int]] = {"zarrs": frozenset({3})}

#: Pipelines whose buffers `tracemalloc` cannot see, for the same reason
#: `NATIVE` exists on an adapter: encoding happens in Rust, and a `0.0` in the
#: peak column would read as 'uses no memory' rather than 'not measurable here'.
NATIVE = frozenset({"zarrs"})


def install(name: str) -> None:
    """Point zarr-python's codec pipeline at `name`, for this whole process.

    Global on purpose. The point of the axis is that no adapter has to cooperate
    -- whichever library the child goes on to import writes through whatever is
    set here, which is exactly the thing being measured.
    """
    path = PIPELINES[name]
    if path is None:
        return
    import zarr

    zarr.config.set({"codec_pipeline.path": path})


def distribution(name: str) -> str:
    """The installed package a pipeline comes from, or an empty string.

    Taken from the configured path rather than listed a second time, so there is
    one place a pipeline is spelled. Recorded per row for the same reason
    `impl_version` is: which zarrs produced a number is part of the number.
    """
    path = PIPELINES.get(name)
    return path.split(".")[0] if path else ""


def unsupported(name: str, zarr_format: int) -> str:
    """Why this pipeline cannot encode this zarr format, or an empty string."""
    formats = FORMATS.get(name)
    if formats is not None and zarr_format not in formats:
        return (
            f"{name} is a zarr v{'/v'.join(str(f) for f in sorted(formats))} "
            f"codec pipeline; this image is v{zarr_format}"
        )
    return ""
