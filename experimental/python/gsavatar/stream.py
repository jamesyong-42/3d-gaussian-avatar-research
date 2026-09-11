"""GsAvatarStream — Building-Blocks analogue (Topic 4).

RecordGsStreamData(lod) / ApplyGsStreamData(bytes). Transport-agnostic payload.
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import (
    AA_SCALE,
    BODY_DIM,
    COORD_Y_UP_LEFT,
    EXPR_FULL,
    EXPR_MEDIUM,
    EXPR_SCALE,
    HAND_DIM,
    HAS_EXPR,
    HAS_FACE,
    HAS_HANDS,
    HAS_ROOT,
    LOD_FULL,
    LOD_HIGH,
    LOD_LOW,
    LOD_MEDIUM,
    LOW_BODY_INDICES,
    PROTOCOL_U8,
    PROTOCOL_VERSION,
    RIG_SMPLX,
    TELEPORT,
)
from .pose import SmplxPose


def _clip_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), -32767, 32767).astype(np.int16)


def quantize_aa(rad: np.ndarray) -> np.ndarray:
    return _clip_i16(np.asarray(rad, dtype=np.float64) * AA_SCALE)


def dequantize_aa(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) / AA_SCALE


def quantize_expr(v: np.ndarray) -> np.ndarray:
    return _clip_i16(np.asarray(v, dtype=np.float64) * EXPR_SCALE)


def dequantize_expr(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) / EXPR_SCALE


def _pack_i16(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype="<i2").tobytes()


def _pack_f16(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype="<f2").tobytes()


def _unpack_i16(data: bytes, n: int, offset: int) -> tuple[np.ndarray, int]:
    raw = data[offset : offset + n * 2]
    if len(raw) != n * 2:
        raise ValueError(f"truncated int16[{n}] at {offset}")
    return np.frombuffer(raw, dtype="<i2").copy(), offset + n * 2


def _unpack_f16(data: bytes, n: int, offset: int) -> tuple[np.ndarray, int]:
    raw = data[offset : offset + n * 2]
    if len(raw) != n * 2:
        raise ValueError(f"truncated f16[{n}] at {offset}")
    return np.frombuffer(raw, dtype="<f2").astype(np.float64), offset + n * 2


@dataclass
class GsAvatarSpawn:
    protocol: int = PROTOCOL_VERSION
    asset_id: bytes = field(default_factory=lambda: uuid.uuid4().bytes)
    rig: int = RIG_SMPLX
    lod_caps: int = 0x0F  # all LODs
    coord_frame: int = COORD_Y_UP_LEFT
    n_gauss: int = 0
    n_bones: int = 0
    n_betas: int = 16
    rest_pose_url: str = ""

    def encode(self) -> bytes:
        url = self.rest_pose_url.encode("utf-8")
        aid = self.asset_id if len(self.asset_id) == 16 else uuid.UUID(bytes=self.asset_id).bytes
        return (
            struct.pack(
                "<H16sBBBIHHH",
                self.protocol,
                aid,
                self.rig,
                self.lod_caps,
                self.coord_frame,
                self.n_gauss,
                self.n_bones,
                self.n_betas,
                len(url),
            )
            + url
        )

    @classmethod
    def decode(cls, data: bytes) -> "GsAvatarSpawn":
        header_size = struct.calcsize("<H16sBBBIHHH")
        if len(data) < header_size:
            raise ValueError("spawn too short")
        proto, aid, rig, lod_caps, coord, n_gauss, n_bones, n_betas, url_len = struct.unpack(
            "<H16sBBBIHHH", data[:header_size]
        )
        url = data[header_size : header_size + url_len].decode("utf-8")
        return cls(
            protocol=proto,
            asset_id=aid,
            rig=rig,
            lod_caps=lod_caps,
            coord_frame=coord,
            n_gauss=n_gauss,
            n_bones=n_bones,
            n_betas=n_betas,
            rest_pose_url=url,
        )


@dataclass
class GsAvatarSnapshot:
    protocol: int = PROTOCOL_U8
    stream_lod: int = LOD_MEDIUM
    seq: int = 0
    capture_tick: int = 0
    flags: int = HAS_ROOT | HAS_FACE | HAS_HANDS | HAS_EXPR
    root_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_rot: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    body: np.ndarray = field(default_factory=lambda: np.zeros(BODY_DIM))
    left_hand: np.ndarray = field(default_factory=lambda: np.zeros(HAND_DIM))
    right_hand: np.ndarray = field(default_factory=lambda: np.zeros(HAND_DIM))
    expr: np.ndarray = field(default_factory=lambda: np.zeros(EXPR_FULL))

    def to_pose(self) -> SmplxPose:
        expr = self.expr
        if self.stream_lod == LOD_MEDIUM:
            padded = np.zeros(EXPR_FULL)
            n = min(EXPR_MEDIUM, expr.size)
            padded[:n] = expr[:n]
            expr = padded
        elif self.stream_lod == LOD_LOW:
            expr = np.zeros(EXPR_FULL)
        return SmplxPose(
            body=self.body,
            left_hand=self.left_hand,
            right_hand=self.right_hand,
            expr=expr,
            root_pos=self.root_pos,
            root_rot=self.root_rot,
            capture_tick=self.capture_tick,
            teleport=bool(self.flags & TELEPORT),
        )

    @classmethod
    def from_pose(cls, pose: SmplxPose, lod: int, seq: int) -> "GsAvatarSnapshot":
        flags = HAS_ROOT
        if lod >= LOD_MEDIUM:
            flags |= HAS_FACE | HAS_HANDS | HAS_EXPR
        elif lod == LOD_HIGH or lod == LOD_FULL:
            flags |= HAS_FACE | HAS_HANDS | HAS_EXPR
        if pose.teleport:
            flags |= TELEPORT
        return cls(
            stream_lod=lod,
            seq=seq,
            capture_tick=int(pose.capture_tick),
            flags=flags,
            root_pos=pose.root_pos.copy(),
            root_rot=pose.root_rot.copy(),
            body=pose.body.copy(),
            left_hand=pose.left_hand.copy(),
            right_hand=pose.right_hand.copy(),
            expr=pose.expr.copy(),
        )


class StreamCodec:
    """RecordGsStreamData / ApplyGsStreamData."""

    @staticmethod
    def record(pose: SmplxPose, lod: int, seq: int) -> bytes:
        return StreamCodec.encode(GsAvatarSnapshot.from_pose(pose, lod, seq))

    @staticmethod
    def apply(data: bytes) -> SmplxPose:
        return StreamCodec.decode(data).to_pose()

    @staticmethod
    def encode(snap: GsAvatarSnapshot) -> bytes:
        lod = int(snap.stream_lod)
        if lod not in (LOD_LOW, LOD_MEDIUM, LOD_HIGH, LOD_FULL):
            raise ValueError(f"bad lod {lod}")
        flags = int(snap.flags)
        buf = bytearray(
            struct.pack(
                "<BBIIH",
                PROTOCOL_U8,
                lod,
                int(snap.seq) & 0xFFFFFFFF,
                int(snap.capture_tick) & 0xFFFFFFFF,
                flags & 0xFFFF,
            )
        )
        if flags & HAS_ROOT:
            pos = np.asarray(snap.root_pos, dtype=np.float64).reshape(3)
            rot = np.asarray(snap.root_rot, dtype=np.float64).reshape(4)
            buf += struct.pack("<7f", *pos.tolist(), *rot.tolist())

        body = np.zeros(BODY_DIM, dtype=np.float64)
        b_in = np.asarray(snap.body, dtype=np.float64).reshape(-1)
        body[: min(BODY_DIM, b_in.size)] = b_in[:BODY_DIM]

        lh = np.zeros(HAND_DIM)
        rh = np.zeros(HAND_DIM)
        li = np.asarray(snap.left_hand, dtype=np.float64).reshape(-1)
        ri = np.asarray(snap.right_hand, dtype=np.float64).reshape(-1)
        lh[: min(HAND_DIM, li.size)] = li[:HAND_DIM]
        rh[: min(HAND_DIM, ri.size)] = ri[:HAND_DIM]

        expr = np.asarray(snap.expr, dtype=np.float64).reshape(-1)

        if lod == LOD_FULL:
            buf += _pack_f16(body)
            if flags & HAS_HANDS:
                buf += _pack_f16(lh)
                buf += _pack_f16(rh)
            if flags & HAS_EXPR:
                e = np.zeros(EXPR_FULL)
                e[: min(EXPR_FULL, expr.size)] = expr[:EXPR_FULL]
                buf += _pack_f16(e)
            return bytes(buf)

        if lod == LOD_LOW:
            buf += _pack_i16(quantize_aa(body[list(LOW_BODY_INDICES)]))
            return bytes(buf)

        buf += _pack_i16(quantize_aa(body))
        if flags & HAS_HANDS:
            buf += _pack_i16(quantize_aa(lh))
            buf += _pack_i16(quantize_aa(rh))
        if flags & HAS_EXPR:
            if lod == LOD_MEDIUM:
                e = np.zeros(EXPR_MEDIUM)
                e[: min(EXPR_MEDIUM, expr.size)] = expr[:EXPR_MEDIUM]
                buf += _pack_i16(quantize_expr(e))
            else:
                e = np.zeros(EXPR_FULL)
                e[: min(EXPR_FULL, expr.size)] = expr[:EXPR_FULL]
                buf += _pack_i16(quantize_expr(e))
        return bytes(buf)

    @staticmethod
    def decode(data: bytes) -> GsAvatarSnapshot:
        if len(data) < 12:
            raise ValueError("snapshot shorter than header")
        proto, lod, seq, tick, flags = struct.unpack_from("<BBIIH", data, 0)
        if proto != PROTOCOL_U8:
            raise ValueError(f"unsupported protocol {proto}")
        off = 12
        root_pos = np.zeros(3)
        root_rot = np.array([0.0, 0.0, 0.0, 1.0])
        if flags & HAS_ROOT:
            if len(data) < off + 28:
                raise ValueError("truncated root")
            vals = struct.unpack_from("<7f", data, off)
            root_pos = np.array(vals[:3], dtype=np.float64)
            root_rot = np.array(vals[3:], dtype=np.float64)
            off += 28

        body = np.zeros(BODY_DIM)
        lh = np.zeros(HAND_DIM)
        rh = np.zeros(HAND_DIM)
        expr = np.zeros(EXPR_FULL)

        if lod == LOD_FULL:
            body, off = _unpack_f16(data, BODY_DIM, off)
            if flags & HAS_HANDS:
                lh, off = _unpack_f16(data, HAND_DIM, off)
                rh, off = _unpack_f16(data, HAND_DIM, off)
            if flags & HAS_EXPR:
                expr, off = _unpack_f16(data, EXPR_FULL, off)
        elif lod == LOD_LOW:
            q, off = _unpack_i16(data, len(LOW_BODY_INDICES), off)
            body[list(LOW_BODY_INDICES)] = dequantize_aa(q)
        else:
            q, off = _unpack_i16(data, BODY_DIM, off)
            body = dequantize_aa(q)
            if flags & HAS_HANDS:
                q, off = _unpack_i16(data, HAND_DIM, off)
                lh = dequantize_aa(q)
                q, off = _unpack_i16(data, HAND_DIM, off)
                rh = dequantize_aa(q)
            if flags & HAS_EXPR:
                n = EXPR_MEDIUM if lod == LOD_MEDIUM else EXPR_FULL
                q, off = _unpack_i16(data, n, off)
                expr[:n] = dequantize_expr(q)

        return GsAvatarSnapshot(
            protocol=proto,
            stream_lod=lod,
            seq=seq,
            capture_tick=tick,
            flags=flags,
            root_pos=root_pos,
            root_rot=root_rot,
            body=body,
            left_hand=lh,
            right_hand=rh,
            expr=expr,
        )


def snapshot_nbytes(lod: int, flags: Optional[int] = None) -> int:
    """Exact encoded size for zeroed pose (used by bandwidth tests)."""
    pose = SmplxPose.identity()
    if flags is None:
        return len(StreamCodec.record(pose, lod, 0))
    snap = GsAvatarSnapshot.from_pose(pose, lod, 0)
    snap.flags = flags
    return len(StreamCodec.encode(snap))
