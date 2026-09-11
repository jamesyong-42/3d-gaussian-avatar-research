"""Convert LHM/LAM compact PLYs to vanilla 248-byte 3DGS (H3)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from .constants import PLY_FLOATS_PER_VERTEX, SH_REST
from .ply import GaussianCloud, PlyError, write_3dgs_ply

PathLike = Union[str, Path]

VANILLA_ORDER = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(SH_REST)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)


def _parse_header(data: bytes) -> tuple[str, bytes, int, list[str], str]:
    marker = b"end_header\n"
    idx = data.find(marker)
    if idx < 0:
        raise PlyError("no end_header")
    header = data[: idx + len(marker)].decode("ascii", errors="replace")
    body = data[idx + len(marker) :]
    n = None
    fmt = None
    props: list[str] = []
    for line in header.splitlines():
        if line.startswith("format "):
            fmt = line.split(" ", 1)[1].strip()
        elif line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property float "):
            props.append(line.split()[-1])
    if n is None or fmt is None:
        raise PlyError("bad header")
    return header, body, n, props, fmt


def convert_to_vanilla(src: PathLike, dst: PathLike) -> GaussianCloud:
    """Pad/truncate SH rest and require xyz, f_dc, opacity, scale, rot."""
    src = Path(src)
    data = src.read_bytes()
    _, body, n, props, fmt = _parse_header(data)
    if not fmt.startswith("binary_little_endian"):
        raise PlyError(f"need binary_little_endian, got {fmt}")
    width = len(props)
    expected = n * width * 4
    if len(body) != expected:
        raise PlyError(f"body {len(body)} != {expected}")
    packed = np.frombuffer(body, dtype="<f4").reshape(n, width)
    col = {name: packed[:, i] for i, name in enumerate(props)}

    def take3(a, b, c, default=0.0) -> np.ndarray:
        if a in col:
            return np.stack([col[a], col[b], col[c]], axis=1).astype(np.float64)
        return np.full((n, 3), default, dtype=np.float64)

    xyz = take3("x", "y", "z")
    normals = take3("nx", "ny", "nz")
    f_dc = take3("f_dc_0", "f_dc_1", "f_dc_2")
    f_rest = np.zeros((n, SH_REST), dtype=np.float64)
    for i in range(SH_REST):
        key = f"f_rest_{i}"
        if key in col:
            f_rest[:, i] = col[key]
    if "opacity" not in col:
        raise PlyError("missing opacity")
    opacity = col["opacity"].astype(np.float64)
    scale = take3("scale_0", "scale_1", "scale_2")
    if "rot_0" not in col:
        raise PlyError("missing rot_*")
    rot = np.stack([col[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    cloud = GaussianCloud(xyz, normals, f_dc, f_rest, opacity, scale, rot)
    write_3dgs_ply(dst, cloud)
    return cloud
