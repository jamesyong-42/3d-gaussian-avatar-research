"""Local vs remote entity flags (Topic 4.4.5 / Meta Preset_Remote analogue)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional

from .pose import SmplxPose


class Preset(IntEnum):
    LOCAL = 0
    REMOTE = 1


class EntityError(RuntimeError):
    pass


IPoseConsumer = Callable[[SmplxPose], None]


@dataclass
class GsAvatarEntity:
    is_local: bool
    preset: Preset
    allow_procedural_anim: bool = False
    tracking_source: Optional[str] = None
    pose_sink: Optional[IPoseConsumer] = None
    visible: bool = False
    last_pose: Optional[SmplxPose] = field(default=None, repr=False)
    spawn_applied: bool = False
    first_snapshot_applied: bool = False

    def __post_init__(self) -> None:
        self.validate_config()

    def validate_config(self) -> None:
        if self.preset == Preset.REMOTE or not self.is_local:
            if self.tracking_source is not None:
                raise EntityError("remote entity must have tracking_source=null")
            if self.allow_procedural_anim:
                raise EntityError("remote entity must have allow_procedural_anim=false")
            if self.is_local:
                raise EntityError("Preset.REMOTE requires is_local=false")

    @classmethod
    def local(cls, pose_sink: Optional[IPoseConsumer] = None, tracking_source: str = "headset") -> "GsAvatarEntity":
        return cls(
            is_local=True,
            preset=Preset.LOCAL,
            allow_procedural_anim=True,
            tracking_source=tracking_source,
            pose_sink=pose_sink,
            visible=True,
            spawn_applied=True,
            first_snapshot_applied=True,
        )

    @classmethod
    def remote(cls, pose_sink: Optional[IPoseConsumer] = None) -> "GsAvatarEntity":
        return cls(
            is_local=False,
            preset=Preset.REMOTE,
            allow_procedural_anim=False,
            tracking_source=None,
            pose_sink=pose_sink,
            visible=False,
        )

    def apply_spawn(self) -> None:
        self.spawn_applied = True
        self._maybe_show()

    def apply_pose(self, pose: SmplxPose) -> None:
        if not self.is_local and not self.spawn_applied:
            raise EntityError("remote must not render until spawn applied")
        self.last_pose = pose
        self.first_snapshot_applied = True
        self._maybe_show()
        if self.pose_sink is not None:
            self.pose_sink(pose)

    def _maybe_show(self) -> None:
        if self.is_local:
            self.visible = True
            return
        # Meta IsLocal=false: will not render until first Stream Data
        self.visible = self.spawn_applied and self.first_snapshot_applied
