"""Fetch only the pinned public LHM++ inference assets; never import vendor code."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

import requests

LAB = Path(__file__).resolve().parents[1]
EXISTING = LAB.parent / "third_party/LHM/pretrained_models"
CHECKPOINT = LAB / "checkpoints/LHMPP-700M-PixelShuffle"
PRIOR = LAB / "checkpoints/LHMPP-Prior"
MODEL_REV = "5f1c4274068e11b93721219618d36594b6087cb6"
PRIOR_REV = "b683c8f68bede4f318b0bb539730b8e6711d30a0"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(item):
    target = Path(item["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != item["bytes"] or digest(target) != item["sha256"]:
            raise ValueError(f"Refusing to overwrite an unverified existing asset: {target}")
        return dict(item, status="already-verified")
    local = EXISTING / item["name"]
    if item.get("reuse") and local.is_file() and local.stat().st_size == item["bytes"] and digest(local) == item["sha256"]:
        shutil.copy2(local, target)
        print(f"REUSED {item['name']}", flush=True)
        return dict(item, status="copied-verified-existing", localSource=str(local))
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < item["bytes"]:
                headers = {"User-Agent": "GaussianAvatarResearch/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with requests.get(item["url"], headers=headers, stream=True, timeout=(25, 60)) as response:
                    response.raise_for_status()
                    resume = offset > 0 and response.status_code == 206
                    if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Invalid resume offset")
                    received = offset if resume else 0
                    reported = time.monotonic()
                    with partial.open("ab" if resume else "wb") as stream:
                        for chunk in response.iter_content(8 << 20):
                            stream.write(chunk)
                            received += len(chunk)
                            if received > item["bytes"]:
                                raise ValueError("Download exceeds pinned file size")
                            if time.monotonic() - reported > 12:
                                print(f"DOWNLOAD {item['name']}: {received / 1e9:.2f}/{item['bytes'] / 1e9:.2f} GB", flush=True)
                                reported = time.monotonic()
            if partial.stat().st_size != item["bytes"] or digest(partial) != item["sha256"]:
                raise ValueError(f"Pinned size/hash mismatch: {item['name']}")
            partial.rename(target)
            print(f"VERIFIED {item['name']}", flush=True)
            return dict(item, status="downloaded-verified")
        except requests.RequestException as error:
            print(f"Retry {attempt + 1}: {item['name']}: {type(error).__name__}", flush=True)
            if attempt == 2:
                raise


def main():
    selected = []
    for name, size, sha in [
        ("config.json", 3103, "c365a99f917e21e42ef7ee1b7b15b71e4a76e815efeded095e4b5657b2bc8cee"),
        ("model.safetensors", 5226465708, "aa0750e7632352c50421c0e041f1e543277ce73a3af7ff26e446399394cac21e"),
    ]:
        selected.append({"name": name, "bytes": size, "sha256": sha, "target": str(CHECKPOINT / name),
                         "url": f"https://modelscope.cn/api/v1/models/Damo_XR_Lab/LHMPP-700M-PixelShuffle/repo?Revision={MODEL_REV}&FilePath={name}"})
    response = requests.get(f"https://huggingface.co/api/models/3DAIGC/LHMPP-Prior/revision/{PRIOR_REV}?blobs=true", timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != PRIOR_REV:
        raise ValueError("Unexpected prior revision")
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        needed = (name.startswith(("human_model_files/smpl/", "human_model_files/smplx/", "human_model_files/flame_assets/", "human_model_files/flame/"))
                  or name in ("human_model_files/flame/FLAME_NEUTRAL.pkl", "human_model_files/smpl_mean_params.npz",
                              "BiRefNet-general-epoch_244.pth", "arcface_resnet18.pth", "dense_sample_points/1_160000.ply",
                              "voxel_grid/cano_1_volume.npz", "voxel_grid/human_prior_constrain.npz", "voxel_grid/voxel_192.pth"))
        sha = (entry.get("lfs") or {}).get("sha256")
        if not needed or name.endswith(".zip") or name.endswith("FLAME_texture.npz"):
            continue
        if not sha:
            # Small metadata/OBJ files have no LFS digest. Hash bytes obtained at
            # the pinned revision and retain that observed digest in the manifest.
            raw = requests.get(f"https://huggingface.co/3DAIGC/LHMPP-Prior/resolve/{PRIOR_REV}/{name}", timeout=30)
            raw.raise_for_status()
            sha = hashlib.sha256(raw.content).hexdigest()
        selected.append({"name": name, "bytes": entry["size"], "sha256": sha, "reuse": True,
                         "target": str(PRIOR / name), "url": f"https://huggingface.co/3DAIGC/LHMPP-Prior/resolve/{PRIOR_REV}/{name}"})
    manifest = {"modelRevision": MODEL_REV, "priorRevision": PRIOR_REV, "files": selected,
                "scope": "Local research only; public hosting is not commercial-use clearance. No credentials or gate acceptance."}
    report = LAB / "reports/lhmpp-assets.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=2) as pool:
        manifest["files"] = list(pool.map(fetch, selected))
    manifest["verified"] = True
    report.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"All {len(selected)} assets verified: {report}", flush=True)


if __name__ == "__main__":
    main()
