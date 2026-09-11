"""Linear blend skinning for canonical Gaussians (Topic 2 product B / LHM runtime).

p' = Σ_k w_ik T_k p_i
Σ' = R(θ) Σ R(θ)^T
"""

from __future__ import annotations

import numpy as np


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues. aa (..., 3) → (..., 3, 3)."""
    aa = np.asarray(aa, dtype=np.float64)
    single = aa.ndim == 1
    if single:
        aa = aa[None, :]
    theta = np.linalg.norm(aa, axis=-1)
    # Avoid divide-by-zero: identity when theta ~ 0
    k = np.zeros_like(aa)
    nz = theta > 1e-12
    k[nz] = aa[nz] / theta[nz, None]
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zeros = np.zeros_like(kx)
    K = np.stack(
        [
            zeros,
            -kz,
            ky,
            kz,
            zeros,
            -kx,
            -ky,
            kx,
            zeros,
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    I = np.eye(3)[None, :, :]
    c = np.cos(theta)[:, None, None]
    s = np.sin(theta)[:, None, None]
    KK = np.matmul(K, K)
    R = I + s * K + (1.0 - c) * KK
    R[~nz] = np.eye(3)
    return R[0] if single else R


def trs_matrix(pos: np.ndarray, rot_aa: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = axis_angle_to_matrix(rot_aa)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """q = (x, y, z, w)."""
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_bone_transforms(
    rest_local_pos: np.ndarray,
    rest_local_aa: np.ndarray,
    posed_local_aa: np.ndarray,
    parents: np.ndarray,
    root_pos: np.ndarray | None = None,
    root_rot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-kinematics rest and posed world matrices.

    rest_local_pos: (B, 3) joint offset in parent space
    rest_local_aa / posed_local_aa: (B, 3)
    parents: (B,) int, -1 = root
    Returns (T_bone, rest_world) each (B, 4, 4) where T_bone = posed_world @ inv(rest_world)
    """
    B = rest_local_pos.shape[0]
    rest_world = np.zeros((B, 4, 4), dtype=np.float64)
    posed_world = np.zeros((B, 4, 4), dtype=np.float64)
    # Inverse-bind is canonical rest. Live root is applied only to posed_world
    # so T_bone = posed @ inv(rest) carries locomotion (otherwise root cancels).
    posed_root_T = np.eye(4)
    if root_pos is not None:
        posed_root_T[:3, 3] = np.asarray(root_pos, dtype=np.float64).reshape(3)
    if root_rot is not None:
        posed_root_T[:3, :3] = quat_to_matrix(root_rot)

    order = _topo(parents)
    for i in order:
        p = int(parents[i])
        Rrest = trs_matrix(rest_local_pos[i], rest_local_aa[i])
        Rposed = trs_matrix(rest_local_pos[i], posed_local_aa[i])
        if p < 0:
            rest_world[i] = Rrest
            posed_world[i] = posed_root_T @ Rposed
        else:
            rest_world[i] = rest_world[p] @ Rrest
            posed_world[i] = posed_world[p] @ Rposed

    inv_rest = np.linalg.inv(rest_world)
    T_bone = posed_world @ inv_rest
    return T_bone, rest_world


def _topo(parents: np.ndarray) -> list[int]:
    B = parents.shape[0]
    children = {i: [] for i in range(B)}
    roots = []
    for i, p in enumerate(parents.tolist()):
        if p < 0:
            roots.append(i)
        else:
            children[int(p)].append(i)
    out: list[int] = []
    stack = list(roots)
    while stack:
        n = stack.pop(0)
        out.append(n)
        stack.extend(children[n])
    if len(out) != B:
        raise ValueError("bone parent graph is not a forest")
    return out


def lbs_points(
    rest_xyz: np.ndarray,
    weights: np.ndarray,
    bone_ids: np.ndarray,
    T_bone: np.ndarray,
) -> np.ndarray:
    """rest_xyz (N,3), weights (N,K), bone_ids (N,K), T_bone (B,4,4) → (N,3)."""
    rest_xyz = np.asarray(rest_xyz, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    bone_ids = np.asarray(bone_ids, dtype=np.int64)
    N, K = weights.shape
    ones = np.ones((N, 1), dtype=np.float64)
    rest_h = np.concatenate([rest_xyz, ones], axis=1)
    T = T_bone[bone_ids]  # (N, K, 4, 4)
    posed_k = np.einsum("nkij,nj->nki", T, rest_h)
    out = np.einsum("nk,nki->ni", weights, posed_k)
    return out[:, :3]


def lbs_cov(
    rest_cov: np.ndarray,
    weights: np.ndarray,
    bone_ids: np.ndarray,
    T_bone: np.ndarray,
) -> np.ndarray:
    """Rotate per-Gaussian covariance by the blended linear part of T."""
    rest_cov = np.asarray(rest_cov, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    bone_ids = np.asarray(bone_ids, dtype=np.int64)
    Rk = T_bone[bone_ids][:, :, :3, :3]  # (N, K, 3, 3)
    R = np.einsum("nk,nkij->nij", weights, Rk)
    # Σ' = R Σ R^T
    tmp = np.matmul(R, rest_cov)
    return np.matmul(tmp, np.transpose(R, (0, 2, 1)))


def scale_quat_to_cov(scale: np.ndarray, rot_xyzw: np.ndarray) -> np.ndarray:
    """3DGS Σ = R S S^T R^T from log-scale and xyzw quaternion."""
    s = np.exp(np.asarray(scale, dtype=np.float64))
    N = s.shape[0]
    S = np.zeros((N, 3, 3), dtype=np.float64)
    S[:, 0, 0] = s[:, 0]
    S[:, 1, 1] = s[:, 1]
    S[:, 2, 2] = s[:, 2]
    R = np.stack([quat_to_matrix(q) for q in rot_xyzw], axis=0)
    RS = np.matmul(R, S)
    return np.matmul(RS, np.transpose(RS, (0, 2, 1)))
