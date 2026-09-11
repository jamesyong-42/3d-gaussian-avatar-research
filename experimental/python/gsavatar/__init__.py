"""GsAvatar prototype runtime — pose, stream, LBS, asset contract."""

from .asset import AssetError, AvatarAsset, validate_folder
from .constants import (
    BODY_DIM,
    HAND_DIM,
    LOD_FULL,
    LOD_HIGH,
    LOD_LOW,
    LOD_MEDIUM,
    PLY_BYTES_PER_VERTEX,
    PROTOCOL_VERSION,
)
from .entity import EntityError, GsAvatarEntity, Preset
from .lbs import lbs_cov, lbs_points, make_bone_transforms
from .loopback import LoopbackLink, foot_slide_metrics
from .ply import PlyError, read_3dgs_ply, write_3dgs_ply
from .pose import FlamePose, SmplxPose
from .stream import GsAvatarSnapshot, GsAvatarSpawn, StreamCodec

__all__ = [
    "AssetError",
    "AvatarAsset",
    "BODY_DIM",
    "EntityError",
    "FlamePose",
    "GsAvatarEntity",
    "GsAvatarSnapshot",
    "GsAvatarSpawn",
    "HAND_DIM",
    "LOD_FULL",
    "LOD_HIGH",
    "LOD_LOW",
    "LOD_MEDIUM",
    "LoopbackLink",
    "PLY_BYTES_PER_VERTEX",
    "PlyError",
    "Preset",
    "PROTOCOL_VERSION",
    "SmplxPose",
    "StreamCodec",
    "foot_slide_metrics",
    "lbs_cov",
    "lbs_points",
    "make_bone_transforms",
    "read_3dgs_ply",
    "validate_folder",
    "write_3dgs_ply",
]
