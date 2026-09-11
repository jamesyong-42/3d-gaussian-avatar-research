from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import BODY_DIM, EXPR_FULL, HAND_DIM


def _vec(data, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"{name} expected {dim} floats, got {arr.size}")
    return arr


@dataclass
class SmplxPose:
    """Canonical pose sink payload (Topic 2 IPoseConsumer.ApplySmplx)."""

    body: np.ndarray  # (75,)
    left_hand: np.ndarray = field(default_factory=lambda: np.zeros(HAND_DIM))
    right_hand: np.ndarray = field(default_factory=lambda: np.zeros(HAND_DIM))
    expr: np.ndarray = field(default_factory=lambda: np.zeros(EXPR_FULL))
    root_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_rot: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    capture_tick: int = 0
    teleport: bool = False

    def __post_init__(self) -> None:
        self.body = _vec(self.body, BODY_DIM, "body")
        self.left_hand = _vec(self.left_hand, HAND_DIM, "left_hand")
        self.right_hand = _vec(self.right_hand, HAND_DIM, "right_hand")
        expr = np.asarray(self.expr, dtype=np.float64).reshape(-1)
        if expr.size not in (0, 3, EXPR_FULL):
            raise ValueError(f"expr expected 0, 3, or {EXPR_FULL} floats, got {expr.size}")
        self.expr = expr
        self.root_pos = _vec(self.root_pos, 3, "root_pos")
        self.root_rot = _vec(self.root_rot, 4, "root_rot")
        n = float(np.linalg.norm(self.root_rot))
        if n < 1e-8:
            self.root_rot = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            self.root_rot = self.root_rot / n

    @classmethod
    def identity(cls, tick: int = 0) -> "SmplxPose":
        return cls(body=np.zeros(BODY_DIM), capture_tick=tick)

    def copy(self) -> "SmplxPose":
        return SmplxPose(
            body=self.body.copy(),
            left_hand=self.left_hand.copy(),
            right_hand=self.right_hand.copy(),
            expr=self.expr.copy(),
            root_pos=self.root_pos.copy(),
            root_rot=self.root_rot.copy(),
            capture_tick=self.capture_tick,
            teleport=self.teleport,
        )


@dataclass
class FlamePose:
    """Canonical face sink (Topic 2 IPoseConsumer.ApplyFlame)."""

    expr: np.ndarray
    jaw: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    neck: np.ndarray = field(default_factory=lambda: np.zeros(3))
    capture_tick: int = 0

    def __post_init__(self) -> None:
        self.expr = np.asarray(self.expr, dtype=np.float64).reshape(-1)
        self.jaw = _vec(self.jaw, 3, "jaw")
        self.rotation = _vec(self.rotation, 3, "rotation")
        self.neck = _vec(self.neck, 3, "neck")
