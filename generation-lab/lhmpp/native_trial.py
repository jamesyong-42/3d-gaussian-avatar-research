"""Run the pinned native LHM++ Gaussian path; portable export is a separate gate."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path("/opt/lhmpp")
OUT = Path("/evidence")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.update(APP_ENABLED="1", APP_MODEL_NAME="LHMPP-700M-PixelShuffle", APP_TYPE="infer.human_lrm_a4o", NUMBA_THREADING_LAYER="omp")
torch._dynamo.config.disable = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def render_views(gs):
    from gsplat import rasterization
    xyz = gs.xyz.float()
    low, high = xyz.amin(0), xyz.amax(0)
    target = (low + high) / 2
    distance = max(3.4, float((high - low)[1]) * 2.2)
    intr = torch.tensor([[760., 0, 252], [0, 760., 420], [0, 0, 1]], device="cuda")
    results = {}
    for name, angle in (("front", 0.), ("side", math.pi/2), ("back", math.pi)):
        eye = target + torch.tensor([math.sin(angle)*distance, 0., math.cos(angle)*distance], device="cuda")
        forward = torch.nn.functional.normalize(target - eye, dim=0)
        right = torch.nn.functional.normalize(torch.linalg.cross(forward, torch.tensor([0.,1.,0.], device="cuda")), dim=0)
        down = torch.linalg.cross(forward, right)
        view = torch.eye(4, device="cuda")
        view[:3, :3] = torch.stack([right, down, forward])
        view[:3, 3] = -view[:3, :3] @ eye
        pixels, alpha, _ = rasterization(xyz, gs.rotation.float(), gs.scaling.float(), gs.opacity.float().reshape(-1),
                                          gs.shs.float().reshape(len(xyz), -1)[:, :3], view[None], intr[None], 504, 840,
                                          backgrounds=torch.ones(1, 3, device="cuda"), packed=False)
        array = (pixels[0].clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
        Image.fromarray(array).save(OUT / f"native-{name}.png")
        results[name] = {"visiblePixels": int((alpha[0] > .1).sum()), "file": f"native-{name}.png"}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    record = {"model": "LHMPP-700M-PixelShuffle", "sourceRevision": "906b5d9fb967ab42efb92f6fa55bf22cac86b653",
              "modelRevision": "5f1c4274068e11b93721219618d36594b6087cb6", "views": args.views, "seed": 42,
              "renderer": "Native Gaussian RGB only; no neural refinement", "status": "starting", "network": "disabled"}
    started = time.perf_counter()
    try:
        record["checkpointSha256"] = sha("/checkpoint/model.safetensors")
        if record["checkpointSha256"] != "aa0750e7632352c50421c0e041f1e543277ce73a3af7ff26e446399394cac21e":
            raise ValueError("Checkpoint mismatch")
        record["checkpointConfigSha256"] = sha("/checkpoint/config.json")
        record["hashSeconds"] = time.perf_counter() - started
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        spec = importlib.util.spec_from_file_location("native_gs", ROOT / "scripts/inference/to_gs_ply.py")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        def checked_model(cfg):
            from core.models import model_dict
            from core.utils.hf_hub import wrap_model_hub
            from safetensors.torch import load_model
            arcface_sha = sha(ROOT / "pretrained_models/arcface_resnet18.pth")
            if arcface_sha != "69554b3184575bdda7ecf7f40caf2b0c5ecbd0617b676d2209d8bf0e7e3d8042":
                raise ValueError("Separate ArcFace prior mismatch")
            config = json.loads(Path(cfg.model_name, "config.json").read_text(encoding="utf-8"))
            instance = wrap_model_hub(model_dict["human_lrm_a4o"])(config)
            missing, unexpected = load_model(instance, str(Path(cfg.model_name, "model.safetensors")), strict=False)
            missing_inference = sorted(k for k in missing if not k.startswith("id_face_net."))
            record["checkpointCoverage"] = {"missingInferenceKeys": missing_inference, "unexpectedKeys": sorted(unexpected),
                                            "separatelyLoadedArcFaceKeys": sorted(missing), "arcfaceSha256": arcface_sha}
            if missing_inference or unexpected:
                raise ValueError(f"Incomplete reconstruction checkpoint: {missing_inference}, unexpected: {unexpected}")
            record["checkpointCoverageVerified"] = True
            # This upstream model overrides eval() without returning self.
            instance.eval()
            return instance
        # ArcFace is deliberately absent from the main checkpoint. Its upstream
        # constructor strictly loads the separately hash-verified prior. Every
        # other missing key, and every unexpected key, remains fatal.
        native.build_app_model = checked_model
        record["checkpointCoverageVerified"] = False
        cli = native.build_arg_parser().parse_args([
            "--model_name", record["model"], "--model_path", "/checkpoint", "--ref_view", str(args.views),
            "--image_glob", str(args.input_dir / "*.png" if args.input_dir else ROOT / "assets/example_multi_images/00000_yuliang_*.png"),
            "--output", str(OUT / "canonical.ply"), "--work_dir", str(OUT / "work")])
        paths = native._resolve_image_paths(cli)
        if args.input_dir and len(paths)!=args.views: raise ValueError("Normalized upload count does not match the requested view count")
        indices = np.linspace(0, len(paths)-1, args.views, dtype=int) if args.views > 1 else [0]
        record["inputs"] = [{"file": Path(paths[i]).name, "sha256": sha(paths[i])} for i in indices]
        record["datasetLimit"] = ("User-uploaded photos; same-person consistency, consent, and held-out identity quality are not automatically verified." if args.input_dir else "Bundled same-subject example sequence; no independent consent/identity-quality audit or calibrated held-out cameras.")
        save_json("metrics.json", record)
        stage = time.perf_counter()
        print("NATIVE loading model", flush=True)
        model, cfg, refs, params, motion, estimator, device = native.setup_loaders_and_inputs(cli)
        model.eval()
        torch.cuda.synchronize()
        record["loadSeconds"] = time.perf_counter() - stage
        record["shapeEstimatorEnabled"] = estimator is not None
        captured = []
        original = model.infer_single_view
        def intercept(*a, **kw):
            result = original(*a, **kw)
            captured[:] = [result]
            return result
        model.infer_single_view = intercept
        stage = time.perf_counter()
        print("NATIVE reconstructing", flush=True)
        try:
            native.run_tpose_export(model, refs, motion, device, str(OUT / "canonical.ply"))
        finally:
            model.infer_single_view = original
        torch.cuda.synchronize()
        record["inferenceAndPlySeconds"] = time.perf_counter() - stage
        result = captured[0]
        attrs, queries, neutral, features = result[:4]
        predicted = result[7] if len(result) == 8 else None
        smplx = {k: v.to(device) for k, v in motion["smplx_params"].items()}
        merged = type(model).smplx_params_with_pred_shape_betas(smplx, predicted)
        record["betas"] = merged["betas"].reshape(-1).detach().cpu().tolist()
        record["shape"] = {"mode": "predicted", "estimator": "LHM++ ShapeHead", "betas": record["betas"]}
        record["gaussians"] = len(attrs[0].offset_xyz)
        record["attributeShapes"] = {k: list(getattr(attrs[0], k).shape) for k in ("offset_xyz", "rotation", "scaling", "opacity", "shs")}
        single = native.build_tpose_smplx_params(native.slice_motion_seq_to_single_frame(motion), neutral, merged["betas"], torch.device(device), torch.float32)
        with torch.no_grad():
            canonical = model.inference_gs(attrs, queries, single, motion["render_c2ws"].to(device), motion["render_intrs"].to(device), motion["render_bg_colors"].to(device), features, pad_forward=False)
            for key in ("xyz", "rotation", "scaling", "opacity", "shs"):
                if not torch.isfinite(getattr(canonical, key)).all():
                    raise ValueError(f"Nonfinite Gaussian {key}")
            record["renders"] = render_views(canonical)
            if any(view["visiblePixels"] < 1000 for view in record["renders"].values()):
                raise ValueError("Native render has too few visible avatar pixels")
        record["status"] = "native-passed"
        record["portable"] = "Not yet exported or validated"
        save_json("metrics.json", record)
        try:
            from export_portable import export_portable
            stage = time.perf_counter()
            print("NATIVE exporting portable bindings and seven reference poses", flush=True)
            meta = export_portable(model, result, merged, OUT, record)
            record.update(portable="exported-awaiting-independent-validation", assetBytes=meta["bytes"],
                          influences=meta["influences"], bindings=meta["generation"]["bindings"],
                          exportSeconds=time.perf_counter()-stage)
        except Exception as export_error:
            record["portable"] = "failed"
            record["portableError"] = f"{type(export_error).__name__}: {export_error}"
            print(record["portableError"], flush=True)
        # Captured neural outputs stay in this process; only a validated portable
        # package may be promoted into the existing browser library.
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(workerSeconds=time.perf_counter()-started, peakAllocatedBytes=torch.cuda.max_memory_allocated(),
                      peakReservedBytes=torch.cuda.max_memory_reserved(), torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name())
        save_json("metrics.json", record)
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
