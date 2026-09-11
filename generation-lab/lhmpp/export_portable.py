"""Bake LHM++'s diffused bindings into the existing linear-blend asset contract.

The export fails if expressions can change non-fixed volumetric query weights.
It must subsequently pass independent Node/browser native-reference checks.
"""
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/portable-backend")
from export_lhm import BinaryAsset, JOINTS
from core.models.rendering.smplx_gsavatar.lbs import blend_shapes


@torch.no_grad()
def export_portable(model, outputs, params, folder: Path, generation):
    attrs, queries, neutral_joint = outputs[:3]
    attr = attrs[0]
    query = queries["neutral_coords"][0]
    renderer, rig = model.renderer, model.renderer.smplx_model
    mean = query + attr.offset_xyz
    n, bones = rig.skinning_weight.shape
    expressions = rig.expr_dirs.shape[-1]
    fixed = (rig.is_rhand + rig.is_lhand + rig.is_face) > 0
    outside = rig.expr_dirs[~fixed]
    max_outside = float(outside.abs().max()) if outside.numel() else 0.
    if max_outside != 0.:
        raise ValueError(f"Expression-dependent volumetric bindings cannot be frozen: outside-mask max {max_outside}")
    weights = rig.query_voxel_skinning_weights(mean.unsqueeze(0))[0].float()
    weights[fixed] = rig.skinning_weight[fixed].float()
    if not torch.isfinite(weights).all() or weights.min() < 0:
        raise ValueError("Invalid diffused skinning weights")
    # No normalization or top-K truncation: preserve the native values exactly.
    neutral = rig.get_transform_mat_vertex(neutral_joint, mean.unsqueeze(0), fixed.unsqueeze(0))[0]
    null_mean = (neutral[:, :3, :3] @ mean[..., None])[..., 0] + neutral[:, :3, 3]
    null_mean += blend_shapes(params["betas"], rig.shape_dirs)[0]
    expr_zero = torch.einsum("nij,nje->nie", neutral[:, :3, :3], rig.expr_dirs)
    joints = rig.get_zero_pose_human(params["betas"], mean.device, None, None)[0]
    to_np = lambda x: x.detach().float().cpu().numpy()
    rows, bone = torch.nonzero(weights, as_tuple=True)
    counts = torch.bincount(rows, minlength=n)
    offsets = torch.cat((torch.zeros(1, device=mean.device, dtype=torch.int64), counts.cumsum(0)))
    out = BinaryAsset()
    out.add("positions", to_np(null_mean))
    out.add("rotations", to_np(attr.rotation)[:, [1, 2, 3, 0]])
    out.add("scales", to_np(attr.scaling))
    out.add("colors", to_np(attr.shs).reshape(n, -1)[:, :3])
    out.add("opacities", to_np(attr.opacity).reshape(n))
    out.add("neutralLinear", to_np(neutral[:, :3, :3]))
    out.add("expressionDirections", to_np(expr_zero).transpose(2, 0, 1))
    out.add("skinOffsets", offsets.cpu().numpy(), "<u4")
    out.add("skinBones", bone.cpu().numpy(), "<u2")
    out.add("skinWeights", to_np(weights[rows, bone]))
    # LHM++ rotates all Gaussians; the original LHM constrained-body exception
    # must not be inherited just because the container format is shared.
    out.add("rotationLocked", np.zeros(n, dtype=np.uint8), "|u1")
    out.add("regions", (rig.is_face.int()+2*rig.is_lhand.int()+4*rig.is_rhand.int()).cpu().numpy(), "|u1")
    out.add("joints", to_np(joints))
    out.add("parents", rig.smplx_layer.parents.cpu().numpy(), "<i4")
    out.add("betas", to_np(params["betas"]).reshape(-1))
    fixtures = []
    for name in ("rest", "arms", "crouch", "hands", "head_jaw", "expression", "turn"):
        angles = np.zeros((bones, 3), np.float32)
        expr = np.zeros(expressions, np.float32)
        trans = [0., 0., 0.]
        if name == "arms": angles[16, 2], angles[17, 2], angles[18, 1] = -.9, .6, -.7
        elif name == "crouch": angles[[1, 2], 0], angles[[4, 5], 0], angles[3, 0] = -.55, 1.05, .2
        elif name == "hands": angles[25:, 2], angles[20, 1], angles[21, 1] = .45, .6, -.4
        elif name == "head_jaw": angles[15, 1], angles[22, 0], angles[23, 1] = .5, .35, .15
        elif name == "expression": expr[0], expr[4], expr[-1] = .8, -.4, .6
        elif name == "turn": angles[0, 1], trans = .7, [.1, .05, -.2]
        p = {k: v.clone() for k, v in params.items()}
        t = torch.tensor(angles, device=mean.device)
        for key, value in {"root_pose": t[0], "body_pose": t[1:22], "jaw_pose": t[22], "leye_pose": t[23], "reye_pose": t[24],
                           "lhand_pose": t[25:40], "rhand_pose": t[40:55], "expr": torch.tensor(expr, device=mean.device),
                           "trans": torch.tensor(trans, device=mean.device)}.items():
            p[key] = value.reshape_as(p[key])
        p["transform_mat_neutral_pose"] = neutral_joint
        single = renderer._get_single_batch_data(renderer.get_single_view_smpl_data(p, 0), 0)
        expected, _ = renderer.animate_gs_model(attr, query, single, mesh_meta=queries["mesh_meta"])
        out.add(f"reference_{name}_positions", to_np(expected[0].xyz))
        out.add(f"reference_{name}_rotations", to_np(expected[0].rotation)[:, [1, 2, 3, 0]])
        fixtures.append({"name": name, "angles": angles.reshape(-1).tolist(), "expression": expr.tolist(),
                         "rootPosition": trans, "rootRotation": [0, 0, 0, 1]})
    generation = {key: generation[key] for key in ("model", "sourceRevision", "modelRevision", "checkpointSha256", "checkpointConfigSha256",
                  "views", "inputs", "seed", "shape", "checkpointCoverageVerified", "checkpointCoverage", "datasetLimit") if key in generation}
    generation = dict(generation, engine="lhmpp", torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), bindings={"type": "diffused-volume-baked-at-predicted-offsets",
                      "expressionOutsideFixedMaskMax": max_outside, "weightSumMaxError": float((weights.sum(-1)-1).abs().max()),
                      "noConstrainedBodyRotationLock": True})
    meta = {"version": 1, "model": "LHMPP-700M-PixelShuffle", "rig": "smplx", "nGaussians": n,
            "nBones": bones, "nExpressions": expressions, "jointNames": JOINTS,
            "coordinates": "SMPL-X right-handed, Y-up, meters; local rotations axis-angle radians; quaternions xyzw",
            "deformation": "lhm-linear-blend-v1", "skinning": "all-nonzero-float32",
            "license": "Research experiment; LHM++ and SMPL-X/FLAME terms require separate review before distribution",
            "referencePoses": fixtures, "influences": int(rows.numel()), "generation": generation,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.save(folder, meta)
    return meta
