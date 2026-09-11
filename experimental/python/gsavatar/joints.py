"""Meta Movement / OpenXR body joints → SMPL-X θ indices.

Headset-unverified: names follow Meta Body Tracking + common Humanoid aliases.
The spike-3 binder fills SmplxPose.body from this table; do not invent extra joints at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from .constants import BODY_DIM
from .pose import SmplxPose

# SMPL-X body_pose joint order (index 0 is global_orient, stored in body[0:3])
SMPL_X_JOINTS = [
    "pelvis",  # global_orient
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def smplx_slice(joint: str) -> slice:
    if joint not in SMPL_X_JOINTS:
        raise KeyError(joint)
    i = SMPL_X_JOINTS.index(joint)
    return slice(i * 3, i * 3 + 3)


# Meta source joint → SMPL-X joint. Many-to-one is allowed; last writer wins unless weighted.
DEFAULT_MAP: list[dict[str, Any]] = [
    {"src": "Hips", "dst": "pelvis", "w": 1.0},
    {"src": "SpineLower", "dst": "spine1", "w": 1.0},
    {"src": "SpineUpper", "dst": "spine2", "w": 1.0},
    {"src": "Chest", "dst": "spine3", "w": 1.0},
    {"src": "Neck", "dst": "neck", "w": 1.0},
    {"src": "Head", "dst": "head", "w": 1.0},
    {"src": "LeftUpperLeg", "dst": "left_hip", "w": 1.0},
    {"src": "LeftLowerLeg", "dst": "left_knee", "w": 1.0},
    {"src": "LeftFootAnkle", "dst": "left_ankle", "w": 1.0},
    {"src": "LeftFootBall", "dst": "left_foot", "w": 1.0},
    {"src": "RightUpperLeg", "dst": "right_hip", "w": 1.0},
    {"src": "RightLowerLeg", "dst": "right_knee", "w": 1.0},
    {"src": "RightFootAnkle", "dst": "right_ankle", "w": 1.0},
    {"src": "RightFootBall", "dst": "right_foot", "w": 1.0},
    {"src": "LeftShoulder", "dst": "left_collar", "w": 1.0},
    {"src": "LeftArmUpper", "dst": "left_shoulder", "w": 1.0},
    {"src": "LeftArmLower", "dst": "left_elbow", "w": 1.0},
    {"src": "LeftHandWrist", "dst": "left_wrist", "w": 1.0},
    {"src": "RightShoulder", "dst": "right_collar", "w": 1.0},
    {"src": "RightArmUpper", "dst": "right_shoulder", "w": 1.0},
    {"src": "RightArmLower", "dst": "right_elbow", "w": 1.0},
    {"src": "RightHandWrist", "dst": "right_wrist", "w": 1.0},
]


def load_map(path: Union[str, Path, None] = None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(x) for x in DEFAULT_MAP]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data["map"])


def retarget_axis_angles(
    src: dict[str, np.ndarray],
    table: list[dict[str, Any]] | None = None,
) -> SmplxPose:
    """src maps Meta joint name → axis-angle (3,). Missing joints stay zero."""
    table = table or DEFAULT_MAP
    body = np.zeros(BODY_DIM)
    for row in table:
        name = row["src"]
        if name not in src:
            continue
        sl = smplx_slice(row["dst"])
        body[sl] = np.asarray(src[name], dtype=np.float64).reshape(3) * float(row.get("w", 1.0))
    return SmplxPose(body=body)


def map_coverage(table: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    table = table or DEFAULT_MAP
    mapped = {row["dst"] for row in table}
    missing = [j for j in SMPL_X_JOINTS if j not in mapped]
    return {
        "n_src": len(table),
        "n_dst_mapped": len(mapped),
        "n_dst_total": len(SMPL_X_JOINTS),
        "missing_smplx": missing,
        "complete": len(missing) == 0,
    }
