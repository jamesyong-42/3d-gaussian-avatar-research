"""Export the actual LHM runtime to a portable browser asset without patching LHM.

Run using third_party/LHM/lhm_env/Scripts/python.exe. This process deliberately
exits after generation so the browser cannot depend on a live neural renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import random
import subprocess

import numpy as np
from generation_config import DEFAULT_MODEL, MODEL_SPECS, checkpoint_path, gpu_lease, sha256_file, validate_betas
from settings import LHM

WEB = Path(__file__).resolve().parents[1]
JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye", "right_eye",
] + [f"{side}_{finger}{joint}" for side in ("left", "right")
     for finger in ("index", "middle", "pinky", "ring", "thumb") for joint in (1, 2, 3)]


def report(stage: str, progress: float, **extra):
    print("GSAVATAR " + json.dumps({"stage": stage, "progress": progress, **extra}), flush=True)


class BinaryAsset:
    def __init__(self):
        self.parts = []
        self.arrays = {}
        self.size = 0

    def add(self, name, value, dtype="<f4"):
        data = np.ascontiguousarray(value, dtype=dtype)
        padding = (-self.size) % 4
        if padding:
            self.parts.append(bytes(padding))
            self.size += padding
        self.arrays[name] = {"offset": self.size, "shape": list(data.shape), "dtype": data.dtype.str, "bytes": data.nbytes}
        self.parts.append(data.tobytes())
        self.size += data.nbytes

    def save(self, folder: Path, meta: dict):
        payload = b"".join(self.parts)
        (folder / "avatar.bin").write_bytes(payload)
        meta.update(arrays=self.arrays, binary="avatar.bin", bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        (folder / "avatar.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def export(runner, captured, folder: Path, model_name=DEFAULT_MODEL, generation=None):
    import torch
    from LHM.models.rendering.smplx_gsavatar.lbs import blend_shapes

    attrs, queries, params = captured
    attr, query = attrs[0], queries[0]
    renderer = runner.model.renderer
    rig = renderer.smplx_model
    device = query.device
    n, bones = rig.skinning_weight.shape
    expressions = rig.expr_dirs.shape[-1]
    as_np = lambda tensor: tensor.detach().float().cpu().numpy()
    weights = rig.skinning_weight.float()
    mean = query + attr.offset_xyz
    mask = ((rig.is_rhand + rig.is_lhand + rig.is_face) > 0).unsqueeze(0)
    neutral = rig.get_transform_mat_vertex(params["transform_mat_neutral_pose"], mean.unsqueeze(0), mask)[0]
    # These are precisely the quantities in transform_to_posed_verts_from_neutral_pose.
    null_mean = (neutral[:, :3, :3] @ mean[..., None])[..., 0] + neutral[:, :3, 3]
    null_mean += blend_shapes(params["betas"], rig.shape_dirs)[0]
    expr_zero = torch.einsum("nij,nje->nie", neutral[:, :3, :3], rig.expr_dirs)
    joints = rig.get_zero_pose_human(params["betas"], device, None, None)[0]

    # Exact sparse storage: retain every nonzero weight, without top-K truncation.
    row, bone = torch.nonzero(weights, as_tuple=True)
    counts = torch.bincount(row, minlength=n)
    offsets = torch.cat((torch.zeros(1, device=device, dtype=torch.int64), counts.cumsum(0)))
    out = BinaryAsset()
    out.add("positions", as_np(null_mean))
    out.add("rotations", as_np(attr.rotation)[:, [1, 2, 3, 0]])
    out.add("scales", as_np(attr.scaling))
    out.add("colors", as_np(attr.shs).reshape(n, -1)[:, :3])
    out.add("opacities", as_np(attr.opacity).reshape(n))
    out.add("neutralLinear", as_np(neutral[:, :3, :3]))
    out.add("expressionDirections", as_np(expr_zero).transpose(2, 0, 1))
    out.add("skinOffsets", offsets.cpu().numpy(), "<u4")
    out.add("skinBones", bone.cpu().numpy(), "<u2")
    out.add("skinWeights", as_np(weights[row, bone]))
    out.add("rotationLocked", rig.is_constrain_body.cpu().numpy(), "|u1")
    out.add("regions", (rig.is_face.to(torch.int32) + 2 * rig.is_lhand.to(torch.int32) + 4 * rig.is_rhand.to(torch.int32)).cpu().numpy(), "|u1")
    out.add("joints", as_np(joints))
    out.add("parents", rig.smplx_layer.parents.cpu().numpy(), "<i4")
    out.add("betas", as_np(params["betas"]).reshape(-1))

    fixtures = []
    for name in ("rest", "arms", "crouch", "hands", "head_jaw", "expression", "turn"):
        angles = np.zeros((bones, 3), dtype=np.float32)
        expr = np.zeros(expressions, dtype=np.float32)
        trans = [0.0, 0.0, 0.0]
        if name == "arms":
            angles[16, 2], angles[17, 2], angles[18, 1] = -0.9, 0.6, -0.7
        elif name == "crouch":
            angles[[1, 2], 0] = -0.55
            angles[[4, 5], 0] = 1.05
            angles[3, 0] = 0.2
        elif name == "hands":
            angles[25:, 2] = 0.45
            angles[20, 1], angles[21, 1] = 0.6, -0.4
        elif name == "head_jaw":
            angles[15, 1], angles[22, 0], angles[23, 1] = 0.5, 0.35, 0.15
        elif name == "expression":
            expr[0], expr[4], expr[-1] = 0.8, -0.4, 0.6
        elif name == "turn":
            angles[0, 1] = 0.7
            trans = [0.1, 0.05, -0.2]
        p = {key: val.clone() for key, val in params.items()}
        t = torch.tensor(angles, device=device)
        for key, value in {"root_pose": t[0], "body_pose": t[1:22], "jaw_pose": t[22],
                           "leye_pose": t[23], "reye_pose": t[24], "lhand_pose": t[25:40],
                           "rhand_pose": t[40:55], "expr": torch.tensor(expr, device=device),
                           "trans": torch.tensor(trans, device=device)}.items():
            p[key] = value.reshape_as(params[key])
        expected = runner.model.animation_infer_gs(attrs, queries, p)
        out.add(f"reference_{name}_positions", as_np(expected.xyz))
        out.add(f"reference_{name}_rotations", as_np(expected.rotation)[:, [1, 2, 3, 0]])
        fixtures.append({"name": name, "angles": angles.reshape(-1).tolist(), "expression": expr.tolist(), "rootPosition": trans, "rootRotation": [0, 0, 0, 1]})
    meta = {
        "version": 1, "model": model_name, "rig": "smplx", "nGaussians": n,
        "nBones": bones, "nExpressions": expressions, "jointNames": JOINTS,
        "coordinates": "SMPL-X right-handed, Y-up, meters; local rotations axis-angle radians; quaternions xyzw",
        "deformation": "lhm-linear-blend-v1", "skinning": "all-nonzero-float32",
        "license": "Research model: see LHM modelcard.md and SMPL-X terms",
        "referencePoses": fixtures, "influences": int(row.numel()),
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation": generation or {},
    }
    out.save(folder, meta)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=[name for name in MODEL_SPECS if name.startswith("LHM-")], default=DEFAULT_MODEL)
    parser.add_argument("--shape-mode", choices=("zero", "estimate", "file"), default="zero")
    parser.add_argument("--betas-json", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait-lock", type=float, default=0)
    args = parser.parse_args()
    if (args.shape_mode == "file") != (args.betas_json is not None):
        parser.error("--shape-mode file requires --betas-json; other modes must not supply it.")
    source, folder = args.image.resolve(), args.output.resolve()
    if not source.is_file():
        parser.error("Source image does not exist.")
    if (folder / "avatar.json").exists() or (folder / "avatar.bin").exists():
        parser.error("Output already contains an avatar. Use a new output directory.")
    if args.betas_json:
        args.betas_json = args.betas_json.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with gpu_lease(timeout=args.wait_lock):
        reconstruct(args, source, folder)


def reconstruct(args, source, folder):
    started = time.perf_counter()
    checkpoint = checkpoint_path(args.model)
    digest = sha256_file(checkpoint / "model.safetensors")
    expected = MODEL_SPECS[args.model].get("sha256")
    if expected and digest != expected:
        raise ValueError("Checkpoint SHA-256 does not match the pinned official model.")
    provenance_seconds = time.perf_counter() - started
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LHM, text=True,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    os.chdir(LHM)
    sys.path.insert(0, str(LHM))
    # Do not let an inherited Gradio setting silently override model provenance.
    os.environ.pop("APP_MODEL_NAME", None)
    os.environ.pop("APP_INFER", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.argv = ["web-export", f"model_name={args.model}", "export_mesh=True"]
    report("Loading reconstruction model", 0.08)
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    from LHM.runners.infer.human_lrm import HumanLRMInferrer
    from LHM.utils.model_download_utils import AutoModelQuery
    original_query = AutoModelQuery.query
    def pinned_query(self, model_name):
        if model_name != args.model:
            raise ValueError("Unexpected reconstruction model request.")
        return str(checkpoint) + os.sep
    # Pin the checkpoint for this process, without editing upstream source or cache refs.
    AutoModelQuery.query = pinned_query
    try:
        runner = HumanLRMInferrer()
    finally:
        AutoModelQuery.query = original_query
    runner.model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    shape_started = time.perf_counter()
    shape = {"mode": args.shape_mode, "betas": [0.0] * 10}
    if args.shape_mode == "estimate":
        report("Estimating body proportions", 0.24)
        result = runner.pose_estimator(str(source))
        if result.beta is None or not result.is_full_body:
            raise ValueError(f"Body-shape estimation rejected the image: {result.msg}")
        shape.update(betas=validate_betas(np.asarray(result.beta).reshape(-1).tolist()),
                     estimator="LHM Multi-HMR multiHMR_896_L", visibleRatio=float(result.ratio),
                     estimatorMessage=result.msg)
    elif args.shape_mode == "file":
        shape_source = json.loads(args.betas_json.read_text(encoding="utf-8"))
        shape.update(betas=validate_betas(shape_source["betas"]), sourceSha256=sha256_file(args.betas_json))
    torch.cuda.synchronize()
    shape_seconds = time.perf_counter() - shape_started
    generation = {"engine": "lhm", "model": args.model, "checkpointRevision": MODEL_SPECS[args.model]["revision"],
                  "checkpointSha256": digest, "checkpointConfigSha256": sha256_file(checkpoint / "config.json"),
                  "sourceRevision": revision, "exporterSha256": sha256_file(Path(__file__).resolve()),
                  "inputSha256": sha256_file(source), "seed": args.seed, "shape": shape,
                  "sourceSize": runner.cfg.source_size, "precision": "float32 (native encoder internals unchanged)",
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(), "offlineWeights": True}
    (folder / "shape.json").write_text(json.dumps(shape, indent=2), encoding="utf-8")
    captured = []
    original = runner.model.animation_infer_gs

    def intercept(attrs, queries, params):
        captured[:] = [attrs, queries, params]
        return original(attrs, queries, params)

    runner.model.animation_infer_gs = intercept
    report("Reconstructing photo", 0.30)
    inference_started = time.perf_counter()
    with torch.no_grad():
        try:
            runner.infer_mesh(str(source), str(folder), str(folder), shape_param=shape["betas"])
        finally:
            runner.model.animation_infer_gs = original
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        if not captured:
            raise RuntimeError("Native model did not produce exportable Gaussian bindings.")
        report("Exporting rig and reference poses", 0.70)
        export_started = time.perf_counter()
        meta = export(runner, captured, folder, args.model, generation)
        torch.cuda.synchronize()
    metrics = {"provenanceSeconds": provenance_seconds, "loadSeconds": load_seconds,
               "shapeSeconds": shape_seconds, "reconstructionAndPlySeconds": inference_seconds,
               "exportAndReferencesSeconds": time.perf_counter() - export_started,
               "workerSeconds": time.perf_counter() - started,
               "peakAllocatedBytes": torch.cuda.max_memory_allocated(),
               "peakReservedBytes": torch.cuda.max_memory_reserved(),
               "assetBytes": meta["bytes"], "nGaussians": meta["nGaussians"], "generation": generation}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    report("Ready", 1.0, nGaussians=meta["nGaussians"], seconds=round(elapsed, 2))


if __name__ == "__main__":
    main()
