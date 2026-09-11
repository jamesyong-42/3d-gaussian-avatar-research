"""Topic 1 → Topic 2 asset folder contract."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .constants import COORD_Y_UP_LEFT, PLY_BYTES_PER_VERTEX, RIG_SMPLX, SYNTH_K
from .ply import GaussianCloud, PlyError, read_3dgs_ply, write_3dgs_ply

PathLike = Union[str, Path]


class AssetError(ValueError):
    pass


@dataclass
class AssetMeta:
    model: str
    n_gauss: int
    n_bones: int
    rig: str = "smplx"
    coord: str = "y_up_l"
    units: str = "m"
    k_skin: int = SYNTH_K
    ply_layout: str = "vanilla_3dgs_248"
    license: str = "research-only"
    asset_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "AssetMeta":
        return cls(
            model=str(data["model"]),
            n_gauss=int(data["n_gauss"]),
            n_bones=int(data["n_bones"]),
            rig=str(data.get("rig", "smplx")),
            coord=str(data.get("coord", "y_up_l")),
            units=str(data.get("units", "m")),
            k_skin=int(data.get("k_skin", SYNTH_K)),
            ply_layout=str(data.get("ply_layout", "vanilla_3dgs_248")),
            license=str(data.get("license", "research-only")),
            asset_id=str(data.get("asset_id", uuid.uuid4())),
        )


@dataclass
class AvatarAsset:
    meta: AssetMeta
    cloud: GaussianCloud
    weights: np.ndarray  # (N, K) float16
    bone_ids: np.ndarray  # (N, K) uint16
    rest_bones: np.ndarray  # (B, 4, 4) float32 inverse-bind or rest world
    betas: np.ndarray  # (n_betas,)

    def save(self, folder: PathLike) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.cloud.n != self.meta.n_gauss:
            raise AssetError("n_gauss mismatch")
        write_3dgs_ply(folder / "gaussians.ply", self.cloud)
        w = np.asarray(self.weights, dtype="<f2")
        b = np.asarray(self.bone_ids, dtype="<u2")
        if w.shape != (self.meta.n_gauss, self.meta.k_skin):
            raise AssetError(f"weights {w.shape} != {(self.meta.n_gauss, self.meta.k_skin)}")
        (folder / "skinning.bin").write_bytes(w.tobytes() + b.tobytes())
        rest = np.asarray(self.rest_bones, dtype="<f4")
        (folder / "rest_bones.bin").write_bytes(rest.tobytes())
        betas = np.asarray(self.betas, dtype="<f4")
        (folder / "shape.betas.bin").write_bytes(betas.tobytes())
        (folder / "meta.json").write_text(json.dumps(self.meta.to_json(), indent=2), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: PathLike) -> "AvatarAsset":
        folder = Path(folder)
        meta = AssetMeta.from_json(json.loads((folder / "meta.json").read_text(encoding="utf-8")))
        cloud = read_3dgs_ply(folder / "gaussians.ply")
        if cloud.n != meta.n_gauss:
            raise AssetError(f"PLY N={cloud.n} meta.n_gauss={meta.n_gauss}")
        blob = (folder / "skinning.bin").read_bytes()
        w_bytes = meta.n_gauss * meta.k_skin * 2
        if len(blob) != w_bytes * 2:
            raise AssetError(f"skinning.bin {len(blob)} bytes, expected {w_bytes * 2}")
        weights = np.frombuffer(blob[:w_bytes], dtype="<f2").reshape(meta.n_gauss, meta.k_skin).astype(np.float64)
        bone_ids = np.frombuffer(blob[w_bytes:], dtype="<u2").reshape(meta.n_gauss, meta.k_skin).astype(np.int64)
        rest = np.frombuffer((folder / "rest_bones.bin").read_bytes(), dtype="<f4")
        expected = meta.n_bones * 16
        if rest.size != expected:
            raise AssetError(f"rest_bones {rest.size} floats, expected {expected}")
        rest_bones = rest.reshape(meta.n_bones, 4, 4).astype(np.float64)
        betas = np.frombuffer((folder / "shape.betas.bin").read_bytes(), dtype="<f4").astype(np.float64)
        return cls(meta, cloud, weights, bone_ids, rest_bones, betas)


def validate_folder(folder: PathLike) -> list[str]:
    """Return a list of problems; empty means pass."""
    folder = Path(folder)
    errors: list[str] = []
    for name in ("meta.json", "gaussians.ply", "skinning.bin", "rest_bones.bin", "shape.betas.bin"):
        if not (folder / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return errors
    try:
        asset = AvatarAsset.load(folder)
    except (AssetError, PlyError, OSError, json.JSONDecodeError, KeyError) as e:
        return [f"load failed: {e}"]
    ply_size = (folder / "gaussians.ply").stat().st_size
    # header + 248 N
    if asset.cloud.vertex_bytes() != asset.meta.n_gauss * PLY_BYTES_PER_VERTEX:
        errors.append("PLY vertex size is not 248 bytes")
    if asset.meta.ply_layout != "vanilla_3dgs_248":
        errors.append(f"ply_layout {asset.meta.ply_layout!r} is not vanilla_3dgs_248")
    if asset.meta.license == "apache-2.0" and "LHM" in asset.meta.model:
        errors.append("LHM weights are CC BY-NC — do not tag the asset apache-2.0")
    _ = ply_size
    return errors
