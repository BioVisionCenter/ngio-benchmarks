"""The store every read operation reads, written by nobody in the running.

Built with plain zarr-python plus hand-written NGFF metadata. That is
deliberate: if the fixture were written by ngio, ngio would be the only
contestant reading a file produced by its own encoder, and any difference in
chunk layout or codec choice would be a difference this suite created rather
than measured. Writing it at the zarr level makes the bytes neutral and makes
`ngio` and `zarr` read *identical* input.

One level only. `compare-io` reads level 0, so a pyramid would be bytes written
and never touched; `compare-create` is where the pyramid is the experiment. A
single-dataset multiscale is valid NGFF, and declaring exactly what exists is
the honest thing for a fixture to do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ngio_benchmarks.core.data import load_source
from ngio_benchmarks.core.images import zarr_compressors

if TYPE_CHECKING:
    from pathlib import Path

    from ngio_benchmarks.core.images import ImageSpec

#: Axis type by name, as NGFF requires it.
_TYPES = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}


def _axes(spec: ImageSpec) -> list[dict[str, str]]:
    """The `axes` entry of the multiscale metadata."""
    axes = []
    for name in spec.axes:
        entry = {"name": name, "type": _TYPES[name]}
        if entry["type"] == "space":
            entry["unit"] = "micrometer"
        elif entry["type"] == "time":
            entry["unit"] = "second"
        axes.append(entry)
    return axes


def _scale(spec: ImageSpec) -> list[float]:
    """The per-axis scale, pixel size on y/x and 1.0 elsewhere."""
    y, x = spec.pixelsize
    return [{"y": y, "x": x}.get(name, 1.0) for name in spec.axes]


def multiscales(spec: ImageSpec) -> dict[str, Any]:
    """NGFF multiscale metadata for a single-level image."""
    return {
        "version": "0.4" if spec.zarr_format == 2 else "0.5",
        "name": spec.name,
        "axes": _axes(spec),
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [{"type": "scale", "scale": _scale(spec)}],
            }
        ],
    }


def store_path(root: Path, spec: ImageSpec) -> Path:
    """Where `spec`'s read fixture lives.

    Named from the spec so every implementation resolves the same path without
    being told it, and so a changed spec lands somewhere new rather than
    silently reusing a stale store.
    """
    return root / f"fixture_{spec.name}_v{spec.zarr_format}.zarr"


def build(root: Path, spec: ImageSpec) -> Path:
    """Create the read fixture if it is not already there, and return it."""
    import zarr

    path = store_path(root, spec)
    marker = path / ".complete"
    if marker.exists():
        return path

    group = zarr.open_group(store=path, mode="w", zarr_format=spec.zarr_format)
    array = group.create_array(
        name="0",
        shape=spec.shape,
        dtype=spec.dtype,
        chunks=spec.chunks,
        shards=spec.shards,
        compressors=zarr_compressors(spec),
    )
    array[...] = load_source(root, spec)

    meta = multiscales(spec)
    if spec.zarr_format == 2:
        group.attrs["multiscales"] = [meta]
    else:
        # NGFF 0.5 nests everything under an `ome` key in the group's
        # attributes; 0.4 puts `multiscales` at the top level.
        group.attrs["ome"] = {"version": "0.5", "multiscales": [meta]}

    # Written last, so an interrupted build is rebuilt rather than reused. The
    # store is shared by every implementation in the run; half of one would
    # produce numbers that look fine.
    marker.write_text("")
    return path
