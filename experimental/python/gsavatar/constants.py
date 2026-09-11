"""Wire format and pose-layout constants.

Keep in lockstep with csharp/GsAvatar.Runtime/Constants.cs and PROTOCOL.md.
"""

from __future__ import annotations

import math

PROTOCOL_VERSION = 1
PROTOCOL_U8 = 1

# StreamLOD
LOD_LOW = 0
LOD_MEDIUM = 1
LOD_HIGH = 2
LOD_FULL = 3
LOD_NAMES = ("Low", "Medium", "High", "Full")

# Snapshot flags
HAS_ROOT = 1 << 0
HAS_FACE = 1 << 1
HAS_HANDS = 1 << 2
TELEPORT = 1 << 3
HAS_EXPR = 1 << 4

# SMPL-X-ish live vector (Mon3tr θ_b ∈ R^75)
# 0:3   global_orient
# 3:66  body_pose (21 joints × 3)
# 66:69 jaw
# 69:72 leye
# 72:75 reye
BODY_DIM = 75
HAND_DIM = 45  # 15 MANO joints × 3
EXPR_FULL = 50
EXPR_MEDIUM = 3  # visemes / jaw-adjacent compact face
EXPR_LOW = 0

# Axis-angle quantisation: int16 covers ±π
AA_SCALE = 32767.0 / math.pi
# Expression / viseme quantisation: int16 / 1000 → ±32.767
EXPR_SCALE = 1000.0

# Low LOD stores these BODY_DIM indices (global, neck, head, wrists)
# neck = body joint 12 → index 3 + 11*3 = 36
# head = body joint 15 → 3 + 14*3 = 45
# L_wrist = joint 20 → 3 + 19*3 = 60
# R_wrist = joint 21 → 3 + 20*3 = 63
LOW_BODY_INDICES = (
    0,
    1,
    2,  # global_orient
    36,
    37,
    38,  # neck
    45,
    46,
    47,  # head
    60,
    61,
    62,  # left wrist
    63,
    64,
    65,  # right wrist
)

HEADER_BYTES = 12  # u8 + u8 + u32 + u32 + u16
ROOT_BYTES = 28  # pos3 f32 + quat4 f32

# Pass/fail from topic 4: Medium snapshot ≤ 1200 B (Photon-class)
MEDIUM_MAX_BYTES = 1200
# Hypothesis H10 uncompressed Medium @ 30 Hz
H10_MAX_MBPS = 0.1

COORD_Y_UP_LEFT = 0
COORD_Z_UP_RIGHT = 1
COORD_SMPLX = 2

RIG_SMPLX = 0
RIG_FLAME = 1
RIG_HYBRID = 2

# Vanilla 3DGS PLY vertex (aras-p / INRIA): 62 × float32
PLY_FLOATS_PER_VERTEX = 62
PLY_BYTES_PER_VERTEX = PLY_FLOATS_PER_VERTEX * 4  # 248
SH_C0 = 0.28209479177387814
SH_REST = 45  # degree-3 rest coefficients

# Synthetic stick-figure
SYNTH_N_BONES = 5
SYNTH_K = 4
