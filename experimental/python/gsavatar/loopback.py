"""One-machine local/remote loopback with fake latency (Topic 4 spike).

Lockstep rule: apply pose only when snapshot.capture_tick == transform.tick,
or when the snapshot bundles the root (HAS_ROOT) and NetworkTransform is disabled.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from .constants import HAS_ROOT, LOD_MEDIUM, TELEPORT
from .entity import GsAvatarEntity
from .lbs import lbs_points, make_bone_transforms
from .pose import SmplxPose
from .stream import GsAvatarSnapshot, StreamCodec


@dataclass
class QueuedPacket:
    enqueue_time: float
    payload: bytes


@dataclass
class LoopbackLink:
    delay_s: float = 0.06
    lod: int = LOD_MEDIUM
    bundle_root: bool = True
    seq: int = 0
    now: float = 0.0
    tick: int = 0
    dt: float = 1.0 / 30.0
    queue: Deque[QueuedPacket] = field(default_factory=deque)
    remote_transform_tick: int = -1
    remote_root: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # If True, remote NetworkTransform is independently interpolated (the bug)
    desync_transform: bool = False
    desync_ticks: int = 5

    def step_time(self) -> None:
        self.now += self.dt
        self.tick += 1

    def record_local(self, local: GsAvatarEntity, pose: SmplxPose) -> bytes:
        pose = pose.copy()
        pose.capture_tick = self.tick
        local.apply_pose(pose)
        payload = StreamCodec.record(pose, self.lod, self.seq)
        self.seq += 1
        self.queue.append(QueuedPacket(enqueue_time=self.now, payload=payload))
        return payload

    def drain_remote(self, remote: GsAvatarEntity) -> Optional[SmplxPose]:
        applied = None
        while self.queue and self.now >= self.queue[0].enqueue_time + self.delay_s:
            pkt = self.queue.popleft()
            snap = StreamCodec.decode(pkt.payload)
            pose = snap.to_pose()
            if self.desync_transform:
                # Apply pose from tick T onto a transform from T-k (foot-slide class)
                self.remote_transform_tick = int(pose.capture_tick) - self.desync_ticks
                # Shift the root as if NetworkTransform lagged
                lag = self.desync_ticks * self.dt
                # Caller supplies speed via pose; we lag root along +Z by speed*lag if we
                # can infer speed from consecutive packets. For the spike, lag Z by a
                # stored velocity estimate: use pose.root itself minus lag along Z
                # only when tests set `desync_along_z`.
                pose = pose.copy()
                pose.root_pos = pose.root_pos.copy()
                pose.root_pos[2] -= getattr(self, "desync_along_z", 0.0) * lag
            else:
                if self.bundle_root and (snap.flags & HAS_ROOT):
                    self.remote_transform_tick = pose.capture_tick
                    self.remote_root = pose.root_pos.copy()
                else:
                    if pose.capture_tick != self.remote_transform_tick:
                        # Buffer: skip apply until transform catches up.
                        # In this sim the transform is the bundled root, so this
                        # path is the explicit fail-closed lockstep.
                        if not (snap.flags & TELEPORT):
                            continue
            remote.apply_pose(pose)
            applied = pose
        return applied


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def slerp_xyzw(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    a = qa / (np.linalg.norm(qa) + 1e-12)
    b = qb / (np.linalg.norm(qb) + 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < 0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return out / (np.linalg.norm(out) + 1e-12)
    theta = np.arccos(dot)
    s = np.sin(theta)
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / s


@dataclass
class StickRig:
    """5-bone Y-up stick figure used by the foot-slide experiment."""

    rest_local_pos: np.ndarray
    rest_local_aa: np.ndarray
    parents: np.ndarray
    rest_xyz: np.ndarray
    weights: np.ndarray
    bone_ids: np.ndarray
    left_foot_index: int
    right_foot_index: int


def make_stick_rig(n_per_bone: int = 40) -> StickRig:
    # Bones: 0 hips, 1 L-upper, 2 L-lower, 3 R-upper, 4 R-lower
    parents = np.array([-1, 0, 1, 0, 4 - 1], dtype=np.int64)
    parents[4] = 3
    rest_local_pos = np.array(
        [
            [0.0, 1.0, 0.0],  # hips (world, parent -1)
            [-0.12, 0.0, 0.0],  # L hip offset
            [0.0, -0.45, 0.0],  # L knee offset
            [0.12, 0.0, 0.0],  # R hip
            [0.0, -0.45, 0.0],  # R knee
        ],
        dtype=np.float64,
    )
    rest_local_aa = np.zeros((5, 3))
    # Sample Gaussians along each bone in rest world (identity FK)
    T, rest_world = make_bone_transforms(rest_local_pos, rest_local_aa, rest_local_aa, parents)
    pts = []
    wts = []
    ids = []
    # bone length along local -Y for legs, hips a small cloud
    for b, length in [(0, 0.12), (1, 0.45), (2, 0.45), (3, 0.45), (4, 0.45)]:
        for i in range(n_per_bone):
            t = i / max(n_per_bone - 1, 1)
            local = np.array([0.0, -length * t, 0.0, 1.0])
            world = rest_world[b] @ local
            pts.append(world[:3])
            # K=4, primary bone + parent
            w = np.zeros(4)
            bid = np.zeros(4, dtype=np.int64)
            bid[0] = b
            w[0] = 0.85 if b != 0 else 1.0
            p = int(parents[b])
            if p >= 0:
                bid[1] = p
                w[1] = 0.15
                w[0] = 0.85
            else:
                w[0] = 1.0
            wts.append(w)
            ids.append(bid)
    rest_xyz = np.stack(pts)
    weights = np.stack(wts)
    bone_ids = np.stack(ids)
    # Left foot = last sample of bone 2
    left_foot_index = n_per_bone * 3 - 1  # bones 0,1,2
    right_foot_index = n_per_bone * 5 - 1
    return StickRig(
        rest_local_pos=rest_local_pos,
        rest_local_aa=rest_local_aa,
        parents=parents,
        rest_xyz=rest_xyz,
        weights=weights,
        bone_ids=bone_ids,
        left_foot_index=left_foot_index,
        right_foot_index=right_foot_index,
    )


def walk_pose(tick: int, dt: float, speed: float = 1.2) -> tuple[SmplxPose, np.ndarray]:
    """Return (SmplxPose, posed_local_aa[5,3]) for the stick rig."""
    t = tick * dt
    phase = 2.0 * np.pi * 1.4 * t  # ~1.4 Hz walk
    posed_aa = np.zeros((5, 3))
    # hips yaw none; legs swing about X
    posed_aa[1, 0] = 0.55 * np.sin(phase)  # L upper
    posed_aa[2, 0] = 0.90 * max(0.0, -np.sin(phase))  # L knee flex on swing
    posed_aa[3, 0] = 0.55 * np.sin(phase + np.pi)
    posed_aa[4, 0] = 0.90 * max(0.0, -np.sin(phase + np.pi))
    pose = SmplxPose.identity(tick=tick)
    pose.root_pos = np.array([0.0, 0.0, speed * t])
    pose.root_rot = np.array([0.0, 0.0, 0.0, 1.0])
    # Stash a few joints into the SMPL-X body slots so the stream carries them
    pose.body[3:6] = posed_aa[1]  # left hip analogue
    pose.body[6:9] = posed_aa[3]  # right hip analogue
    pose.body[12:15] = posed_aa[2]  # left knee analogue
    pose.body[15:18] = posed_aa[4]
    return pose, posed_aa


def posed_feet(rig: StickRig, posed_aa: np.ndarray, root_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, _ = make_bone_transforms(
        rig.rest_local_pos,
        rig.rest_local_aa,
        posed_aa,
        rig.parents,
        root_pos=root_pos,
    )
    xyz = lbs_points(rig.rest_xyz, rig.weights, rig.bone_ids, T)
    return xyz[rig.left_foot_index], xyz[rig.right_foot_index]


def foot_slide_metrics(
    n_ticks: int = 180,
    delay_s: float = 0.06,
    desync: bool = False,
    speed: float = 1.2,
    dt: float = 1.0 / 30.0,
) -> dict:
    """Walk forward; compare local vs remote left-foot world position.

    Lockstep (desync=False) should keep max error under 1 cm.
    Desync transform (desync=True) should exceed 5 cm — the bug we refuse to ship.
    """
    rig = make_stick_rig()
    local = GsAvatarEntity.local()
    remote = GsAvatarEntity.remote()
    remote.apply_spawn()
    link = LoopbackLink(delay_s=delay_s, dt=dt, desync_transform=desync)
    if desync:
        link.desync_along_z = speed
        link.desync_ticks = 5

    errors = []
    local_feet = []
    remote_feet = []
    last_remote_pose = None
    last_remote_aa = None

    # Need posed_aa on the remote side too. Reconstruct from body slots.
    for _ in range(n_ticks):
        pose, posed_aa = walk_pose(link.tick, dt, speed=speed)
        lf, _ = posed_feet(rig, posed_aa, pose.root_pos)
        local_feet.append(lf)
        link.record_local(local, pose)
        applied = link.drain_remote(remote)
        if applied is not None:
            last_remote_pose = applied
            # rebuild aa from streamed body slots
            last_remote_aa = np.zeros((5, 3))
            last_remote_aa[1] = applied.body[3:6]
            last_remote_aa[3] = applied.body[6:9]
            last_remote_aa[2] = applied.body[12:15]
            last_remote_aa[4] = applied.body[15:18]
        if last_remote_pose is not None:
            rf, _ = posed_feet(rig, last_remote_aa, last_remote_pose.root_pos)
            remote_feet.append(rf)
            # compare against local foot at the snapshot tick, not "now"
            src_tick = last_remote_pose.capture_tick
            if 0 <= src_tick < len(local_feet):
                errors.append(float(np.linalg.norm(rf - local_feet[src_tick])))
            else:
                errors.append(float(np.linalg.norm(rf - lf)))
        link.step_time()

    if not errors:
        raise RuntimeError("loopback produced no remote poses — delay too high?")
    arr = np.array(errors)
    return {
        "n": int(arr.size),
        "mean_m": float(arr.mean()),
        "max_m": float(arr.max()),
        "p95_m": float(np.percentile(arr, 95)),
        "desync": desync,
        "delay_s": delay_s,
        "remote_visible": remote.visible,
        "n_packets_queued": len(link.queue),
    }
