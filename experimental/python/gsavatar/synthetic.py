"""Deterministic stick-figure Gaussian avatar for ingest / LBS / loopback tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .asset import AssetMeta, AvatarAsset
from .constants import SYNTH_K, SYNTH_N_BONES
from .lbs import make_bone_transforms
from .loopback import make_stick_rig
from .ply import GaussianCloud, logit, rgb_to_dc


def build_synthetic_asset(n_per_bone: int = 40, seed: int = 0) -> AvatarAsset:
    rng = np.random.default_rng(seed)
    rig = make_stick_rig(n_per_bone=n_per_bone)
    n = rig.rest_xyz.shape[0]
    rgb = np.tile(np.array([0.72, 0.54, 0.44]), (n, 1))
    rgb += rng.normal(0, 0.02, rgb.shape)
    rgb = np.clip(rgb, 0, 1)
    f_dc = rgb_to_dc(rgb)
    f_rest = np.zeros((n, 45), dtype=np.float64)
    opacity = logit(np.full(n, 0.85))
    scale = np.log(np.full((n, 3), 0.018))
    # 3DGS PLY rot is (w, x, y, z) as rot_0..3 in the official INRIA exporter
    rot = np.zeros((n, 4))
    rot[:, 0] = 1.0
    normals = np.zeros((n, 3))
    cloud = GaussianCloud(
        xyz=rig.rest_xyz.astype(np.float64),
        normals=normals,
        f_dc=f_dc,
        f_rest=f_rest,
        opacity=opacity,
        scale=scale,
        rot=rot,
    )
    T, rest_world = make_bone_transforms(
        rig.rest_local_pos, rig.rest_local_aa, rig.rest_local_aa, rig.parents
    )
    inv_bind = np.linalg.inv(rest_world)
    w = np.zeros((n, SYNTH_K), dtype=np.float64)
    bids = np.zeros((n, SYNTH_K), dtype=np.int64)
    w[:, : rig.weights.shape[1]] = rig.weights
    bids[:, : rig.bone_ids.shape[1]] = rig.bone_ids
    meta = AssetMeta(
        model="synthetic-stick-v1",
        n_gauss=n,
        n_bones=SYNTH_N_BONES,
        rig="smplx",
        coord="y_up_l",
        k_skin=SYNTH_K,
        license="cc0-synthetic",
        asset_id="00000000-0000-0000-0000-000000000001",
    )
    return AvatarAsset(
        meta=meta,
        cloud=cloud,
        weights=w,
        bone_ids=bids,
        rest_bones=inv_bind,
        betas=np.zeros(16, dtype=np.float64),
    )


def write_synthetic(folder: Path, n_per_bone: int = 40) -> Path:
    asset = build_synthetic_asset(n_per_bone=n_per_bone)
    return asset.save(folder)
