"""Vanilla 3DGS PLY (INRIA / aras-p UnityGaussianSplatting).

62 float32 fields = 248 bytes/vertex. LHM/LAM compact PLYs must be converted
here before Unity import (hypothesis H3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from .constants import PLY_BYTES_PER_VERTEX, PLY_FLOATS_PER_VERTEX, SH_C0, SH_REST

PathLike = Union[str, Path]

_REST_PROPS = "".join(f"property float f_rest_{i}\n" for i in range(SH_REST))

HEADER = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {n}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property float nx\n"
    "property float ny\n"
    "property float nz\n"
    "property float f_dc_0\n"
    "property float f_dc_1\n"
    "property float f_dc_2\n"
    f"{_REST_PROPS}"
    "property float opacity\n"
    "property float scale_0\n"
    "property float scale_1\n"
    "property float scale_2\n"
    "property float rot_0\n"
    "property float rot_1\n"
    "property float rot_2\n"
    "property float rot_3\n"
    "end_header\n"
)


class PlyError(ValueError):
    pass


@dataclass
class GaussianCloud:
    xyz: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3)
    f_dc: np.ndarray  # (N, 3)
    f_rest: np.ndarray  # (N, 45)
    opacity: np.ndarray  # (N,) logit
    scale: np.ndarray  # (N, 3) log-scale
    rot: np.ndarray  # (N, 4) 3DGS order wxyz in file (rot_0..3)

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    def vertex_bytes(self) -> int:
        return self.n * PLY_BYTES_PER_VERTEX


def rgb_to_dc(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return (rgb - 0.5) / SH_C0


def dc_to_rgb(dc: np.ndarray) -> np.ndarray:
    return np.asarray(dc, dtype=np.float64) * SH_C0 + 0.5


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def write_3dgs_ply(path: PathLike, cloud: GaussianCloud) -> None:
    path = Path(path)
    n = cloud.n
    parts = [
        np.asarray(cloud.xyz, dtype="<f4"),
        np.asarray(cloud.normals, dtype="<f4"),
        np.asarray(cloud.f_dc, dtype="<f4"),
        np.asarray(cloud.f_rest, dtype="<f4"),
        np.asarray(cloud.opacity, dtype="<f4").reshape(n, 1),
        np.asarray(cloud.scale, dtype="<f4"),
        np.asarray(cloud.rot, dtype="<f4"),
    ]
    for p in parts:
        if p.shape[0] != n:
            raise PlyError(f"field {p.shape} does not match N={n}")
    packed = np.concatenate([p.reshape(n, -1) for p in parts], axis=1)
    if packed.shape[1] != PLY_FLOATS_PER_VERTEX:
        raise PlyError(f"expected {PLY_FLOATS_PER_VERTEX} floats, got {packed.shape[1]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(HEADER.format(n=n).encode("ascii"))
        f.write(packed.astype("<f4").tobytes(order="C"))


def read_3dgs_ply(path: PathLike) -> GaussianCloud:
    path = Path(path)
    data = path.read_bytes()
    marker = b"end_header\n"
    idx = data.find(marker)
    if idx < 0:
        raise PlyError("no end_header")
    header = data[: idx + len(marker)].decode("ascii", errors="replace")
    body = data[idx + len(marker) :]
    n = None
    fmt = None
    props = []
    for line in header.splitlines():
        if line.startswith("format "):
            fmt = line.split(" ", 1)[1].strip()
        elif line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property float "):
            props.append(line.split()[-1])
    if n is None:
        raise PlyError("missing element vertex")
    if fmt != "binary_little_endian 1.0":
        raise PlyError(f"unsupported format {fmt!r}; convert to binary_little_endian")
    if len(props) != PLY_FLOATS_PER_VERTEX:
        raise PlyError(
            f"vertex has {len(props)} float properties ({body.__len__() // max(n, 1)} bytes/vertex); "
            f"vanilla 3DGS / UnityGaussianSplatting expects {PLY_FLOATS_PER_VERTEX} floats "
            f"({PLY_BYTES_PER_VERTEX} bytes). This is H3: LHM/LAM PLYs need a converter."
        )
    expected = n * PLY_BYTES_PER_VERTEX
    if len(body) != expected:
        raise PlyError(f"body {len(body)} bytes, expected {expected} ({PLY_BYTES_PER_VERTEX}×{n})")
    packed = np.frombuffer(body, dtype="<f4").reshape(n, PLY_FLOATS_PER_VERTEX)
    i = 0

    def take(k: int) -> np.ndarray:
        nonlocal i
        sl = packed[:, i : i + k].copy()
        i += k
        return sl

    xyz = take(3)
    normals = take(3)
    f_dc = take(3)
    f_rest = take(SH_REST)
    opacity = take(1).reshape(n)
    scale = take(3)
    rot = take(4)
    return GaussianCloud(xyz, normals, f_dc, f_rest, opacity, scale, rot)


def is_vanilla_layout(path: PathLike) -> tuple[bool, str]:
    try:
        cloud = read_3dgs_ply(path)
        return True, f"vanilla 248-byte 3DGS, N={cloud.n}"
    except PlyError as e:
        return False, str(e)
