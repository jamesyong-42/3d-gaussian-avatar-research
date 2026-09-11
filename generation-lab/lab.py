"""Reproducible local generation experiments. No foundation-model training."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

LAB = Path(__file__).resolve().parent
WEB = LAB.parent / "web"
ROOT = LAB.parent
sys.path.insert(0, str(WEB / "backend"))
from generation_config import LHM, MODEL_SPECS, checkpoint_path, sha256_file
from settings import DATA, GENERATION_PYTHON, generation_environment

PYTHON = GENERATION_PYTHON
CREATE_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CASES = [("hat-skirt", "000057.png", "Hat, loose sleeves, skirt, and bag"),
         ("dark-outfit", "11.JPG", "Stylized dark clothing and face/hand occlusion"),
         ("grayscale", "14.JPG", "Grayscale image, light shirt, trousers, indoor background")]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze():
    path = LAB / "benchmark/manifest.json"
    if path.exists():
        print(f"Already frozen: {path}")
        return
    from PIL import Image
    cases = []
    for case_id, filename, note in CASES:
        source = LHM / "train_data/example_imgs" / filename
        with Image.open(source) as photo:
            size = list(photo.size)
        cases.append({"id": case_id, "source": source.relative_to(ROOT).as_posix(),
                      "sha256": sha256_file(source), "size": size, "note": note,
                      "split": "smoke", "heldOutViews": [],
                      "rights": "Bundled upstream example; no independent consent audit or product-use clearance."})
    baseline = DATA / "avatars/example"
    save(path, {"version": 1, "createdAt": now(), "purpose": "Engineering smoke tests, NOT held-out identity validation",
                "sourceRepo": "https://github.com/aigc3d/LHM", "cases": cases,
                "baseline": {name: sha256_file(baseline / name) for name in ("avatar.json", "avatar.bin")},
                "normalization": "EXIF transpose, RGB, thumbnail within 2048x2048, PNG (same as HTTP upload)",
                "seed": 42, "qualityMetrics": "No calibrated holdouts: do not report PSNR/SSIM/LPIPS or quality winner."})
    print(f"Frozen {len(cases)} inputs: {path}")


def fetch_model(model):
    from huggingface_hub import snapshot_download
    spec = MODEL_SPECS[model]
    print(f"Fetching {model} @ {spec['revision']}", flush=True)
    folder = snapshot_download(repo_id=f"3DAIGC/{model}", revision=spec["revision"],
                               local_dir=str(spec["path"]), allow_patterns=["config.json", "model.safetensors", "README.md"],
                               token=False, max_workers=2)
    digest = sha256_file(Path(folder) / "model.safetensors")
    if spec.get("sha256") and digest != spec["sha256"]:
        raise ValueError("Downloaded checkpoint hash mismatch")
    print(json.dumps({"model": model, "revision": spec["revision"], "sha256": digest, "path": folder}))


def readiness():
    import requests
    endpoints = {
        "lhm-1b": "https://huggingface.co/api/models/3DAIGC/LHM-1B-HF?blobs=true",
        "lhmpp-base": "https://huggingface.co/api/models/3DAIGC/LHMPP-700M?blobs=true",
        "lhmpp-pixelshuffle": "https://huggingface.co/api/models/3DAIGC/LHMPP-700M-PixelShuffle?blobs=true",
        "lhmpp-smplx-free": "https://huggingface.co/api/models/3DAIGC/LHMPP-700M-SMPLX-FREE?blobs=true",
        "idol": "https://huggingface.co/api/models/yiyuzhuang/IDOL?blobs=true",
        "lhmpp-source": "https://api.github.com/repos/aigc3d/LHM-plusplus/commits/main",
        "idol-source": "https://api.github.com/repos/yiyuzhuang/IDOL/commits/main",
        "digs-source": "https://api.github.com/repos/KLMAV-CUC/DiGS-Avatar/commits/main",
    }
    def inspect(item):
        name, url = item
        result = {"id": name, "url": url, "checkedAt": now()}
        try:
            response = requests.get(url, timeout=25, headers={"User-Agent": "GaussianAvatarResearch/1.0"})
            result["httpStatus"] = response.status_code
            if response.ok:
                info = response.json()
                result.update(revision=info.get("sha"), gated=info.get("gated"),
                              license=(info.get("cardData") or {}).get("license"),
                              files=[{"name": f["rfilename"], "bytes": f.get("size"),
                                      "sha256": (f.get("lfs") or {}).get("sha256")} for f in info.get("siblings", [])])
            else:
                result["note"] = "Not publicly accessible without authentication; does not prove it does not exist."
        except requests.RequestException as error:
            result["error"] = str(error)
        print(f"{name}: {result.get('httpStatus', 'network error')}", flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inspect, endpoints.items()))
    payload = {"checkedAt": now(), "diskFreeBytes": shutil.disk_usage(LAB).free, "candidates": results,
               "note": "Metadata only. No weights loaded, accounts used, or licenses accepted by this check."}
    save(LAB / "reports/readiness.json", payload)


def run(args):
    from PIL import Image, ImageOps
    manifest_path = LAB / "benchmark/manifest.json"
    if not manifest_path.exists():
        raise ValueError("Run freeze first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = args.models
    for model in models:
        checkpoint_path(model)
    cases = [c for c in manifest["cases"] if not args.cases or c["id"] in args.cases]
    if not cases:
        raise ValueError("No benchmark cases selected")
    batch_id = "lab-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    batch = LAB / "runs" / batch_id
    batch.mkdir(parents=True)
    results = []
    fitted_shapes = {}
    settings = {"batchId": batch_id, "createdAt": now(), "models": models, "shapes": args.shapes,
                "repeats": args.repeats, "manifestSha256": sha256_file(manifest_path), "results": results}
    env = generation_environment()
    for model in models:
        for case in cases:
            source = ROOT / case["source"]
            if sha256_file(source) != case["sha256"]:
                raise ValueError(f"Frozen source changed: {source}")
            for shape in args.shapes:
                for repeat in range(args.repeats):
                    short_model = "1b" if model == "LHM-1B-HF" else "500m"
                    asset_id = f"{batch_id}-{short_model}-{case['id']}-{shape}-{repeat}"
                    output = DATA / "avatars" / asset_id
                    output.mkdir(parents=True, exist_ok=False)
                    with Image.open(source) as photo:
                        image = ImageOps.exif_transpose(photo).convert("RGB")
                        image.thumbnail((2048, 2048))
                        image.save(output / "input.png")
                        image.thumbnail((256, 256))
                        image.save(output / "thumbnail.jpg", quality=85)
                    shape_mode = "file" if shape == "estimate" and case["id"] in fitted_shapes else shape
                    cmd = [str(PYTHON), "-u", str(WEB / "backend/export_lhm.py"), "--image", str(output / "input.png"),
                           "--output", str(output), "--model", model, "--shape-mode", shape_mode, "--seed", "42", "--wait-lock", "120"]
                    if shape_mode == "file":
                        cmd += ["--betas-json", str(fitted_shapes[case["id"]])]
                    result = {"assetId": asset_id, "case": case["id"], "model": model,
                              "shape": shape, "actualShapeMode": shape_mode, "repeat": repeat,
                              "output": output.relative_to(ROOT).as_posix(), "command": cmd}
                    print(f"RUN {asset_id}", flush=True)
                    started = time.perf_counter()
                    peak_device = 0
                    with (output / "generation.log").open("w", encoding="utf-8") as log:
                        process = subprocess.Popen(cmd, cwd=LHM, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   creationflags=CREATE_FLAGS)
                        while process.poll() is None:
                            try:
                                info = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                               text=True, timeout=5, creationflags=CREATE_FLAGS)
                                peak_device = max(peak_device, int(info.splitlines()[0]))
                            except (OSError, ValueError, subprocess.SubprocessError):
                                pass
                            if time.perf_counter() - started > args.timeout:
                                process.terminate()
                                process.wait(timeout=20)
                                result["timeout"] = True
                                break
                            time.sleep(0.5)
                    result.update(exitCode=process.returncode, wallSeconds=round(time.perf_counter() - started, 3),
                                  peakDeviceMiB=peak_device, deviceMeasurement="Whole GPU, sampled ~0.5s; includes other applications.")
                    if process.returncode == 0 and (output / "avatar.json").exists():
                        meta = json.loads((output / "avatar.json").read_text(encoding="utf-8"))
                        meta["label"] = f"{case['id']} / {short_model} / {shape}"
                        save(output / "avatar.json", meta)
                        result["metrics"] = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                        check = subprocess.run(["node", str(LAB / "validate_asset.mjs"), str(output)],
                                               capture_output=True, text=True, creationflags=CREATE_FLAGS)
                        try:
                            result["validation"] = json.loads(check.stdout)
                        except ValueError:
                            result["validation"] = {"pass": False, "error": check.stderr[-3000:]}
                        result["status"] = "passed" if check.returncode == 0 else "parity-failed"
                        if shape == "estimate":
                            fitted_shapes.setdefault(case["id"], output / "shape.json")
                    else:
                        result["status"] = "failed"
                        result["errorTail"] = (output / "generation.log").read_text(encoding="utf-8", errors="replace")[-4500:]
                    results.append(result)
                    save(batch / "results.json", settings)
                    print(f"{result['status']}: {result['wallSeconds']}s, device peak {peak_device} MiB", flush=True)
    baseline = DATA / "avatars/example"
    settings["baselineUnchanged"] = all(sha256_file(baseline / name) == digest for name, digest in manifest["baseline"].items())
    settings["finishedAt"] = now()
    save(batch / "results.json", settings)
    print(str(batch / "results.json"), flush=True)
    return 0 if settings["baselineUnchanged"] and all(r["status"] == "passed" for r in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("readiness")
    fetch = sub.add_parser("fetch-lhm")
    fetch.add_argument("model", choices=[name for name in MODEL_SPECS if name.startswith("LHM-")])
    experiment = sub.add_parser("run")
    experiment.add_argument("--models", nargs="+", choices=[name for name in MODEL_SPECS if name.startswith("LHM-")], default=["LHM-500M-HF", "LHM-1B-HF"])
    experiment.add_argument("--shapes", nargs="+", choices=("zero", "estimate"), default=["zero", "estimate"])
    experiment.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES])
    experiment.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    experiment.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.command == "freeze": freeze()
    elif args.command == "readiness": readiness()
    elif args.command == "fetch-lhm": fetch_model(args.model)
    else: return run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
