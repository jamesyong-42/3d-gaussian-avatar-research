"""Pinned local reconstruction choices. Importable without CUDA/model dependencies."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time
import subprocess
import shutil
from settings import WEB, LHM, LAB, DATA, DOCKER, DOCKER_RESERVE, GENERATION_ENABLED, GENERATION_PYTHON, configured_path

DEFAULT_MODEL = "LHM-500M-HF"
DEFAULT_SHAPE = "zero"
LHMPP_MODEL = "LHMPP-700M-PixelShuffle"
LHMPP_IMAGE = "gaussian-lhmpp:cu121-source906b5d9"
LHMPP_IMAGE_ID = "sha256:e079244143dabea62b02de368403782b1b755d241c1f2c7534181c315770303a"
LHMPP_VOLUME = "gaussian-lhmpp-py310-env"
MODEL_SPECS = {
    DEFAULT_MODEL: {
        "revision": "dd6392905187a91fd67b3f6962aa74481e943764",
        "path": configured_path("GSAVATAR_LHM_500M_DIR", WEB.parent / "local/models/LHM-500M-HF"),
        "label": "LHM 500M · baseline",
        "shapeModes": ["zero", "estimate"], "maxPhotos": 1,
    },
    "LHM-1B-HF": {
        "revision": "92372582f660066b9f1b9513860744357265b3d5",
        "path": configured_path("GSAVATAR_LHM_1B_DIR", LAB / "checkpoints/LHM-1B-HF"),
        "label": "LHM 1B · experimental",
        "bytes": 6852462632,
        "sha256": "59dc25167d1d72d57fb068445b96e2343ab550b649e9999765200502d03171b9",
        "shapeModes": ["zero", "estimate"], "maxPhotos": 1,
    },
    LHMPP_MODEL: {
        "revision": "5f1c4274068e11b93721219618d36594b6087cb6",
        "path": configured_path("GSAVATAR_LHMPP_DIR", LAB / "checkpoints/LHMPP-700M-PixelShuffle"),
        "label": "LHM++ · 1–8 photos · experimental",
        "bytes": 5226465708,
        "sha256": "aa0750e7632352c50421c0e041f1e543277ce73a3af7ff26e446399394cac21e",
        "shapeModes": ["predicted"], "maxPhotos": 8,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_betas(values) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 10:
        raise ValueError("Body shape must contain exactly ten finite numeric betas.")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
        raise ValueError("Body shape must contain exactly ten finite numeric betas.")
    return [float(x) for x in values]


def validate_choice(model: str, shape: str):
    if model not in MODEL_SPECS:
        raise ValueError("Unsupported reconstruction model.")
    if shape not in MODEL_SPECS[model]["shapeModes"]:
        raise ValueError("Unsupported body-shape mode for this model.")


def checkpoint_path(model: str) -> Path:
    if model not in MODEL_SPECS:
        raise ValueError("Unsupported reconstruction model.")
    spec = MODEL_SPECS[model]
    folder = spec["path"]
    weights = folder / "model.safetensors"
    if not weights.is_file() or not (folder / "config.json").is_file():
        raise FileNotFoundError(f"{model} is not installed. See docs/generation.md for the separate model setup.")
    if spec.get("bytes") and weights.stat().st_size != spec["bytes"]:
        raise ValueError(f"{model} checkpoint is incomplete.")
    return folder


_lhmpp_health = (0, False, "Not checked")


def lhmpp_image_info():
    # Docker Desktop's local tag lookup intermittently reports this retained
    # manifest-list image missing. Its immutable content ID resolves reliably.
    # A newly rebuilt image can still be used via the tag after label validation.
    flags=getattr(subprocess,"CREATE_NO_WINDOW",0)
    for ref in (LHMPP_IMAGE_ID,LHMPP_IMAGE):
        result=subprocess.run([DOCKER,"image","inspect",ref],capture_output=True,text=True,timeout=3,creationflags=flags)
        if result.returncode: continue
        info=json.loads(result.stdout)[0];labels=info.get("Config",{}).get("Labels",{}) or {}
        if labels.get("research.task")!="lhmpp" or labels.get("research.source")!="906b5d9fb967ab42efb92f6fa55bf22cac86b653":
            raise RuntimeError("Unexpected LHM++ image/source label")
        return info
    raise RuntimeError("Start Docker Desktop and install the isolated LHM++ image first.")


def lhmpp_readiness(refresh=False):
    """Read-only, bounded runtime probe. Never starts Docker or installs software."""
    global _lhmpp_health
    if not refresh and time.monotonic()-_lhmpp_health[0]<15: return _lhmpp_health[1:]
    try:
        checkpoint_path(LHMPP_MODEL)
        for name in ("voxel_grid/cano_1_volume.npz", "dense_sample_points/1_160000.ply", "arcface_resnet18.pth", "human_model_files/flame/2019/generic_model.pkl"):
            if not (LAB/"checkpoints/LHMPP-Prior"/name).is_file(): raise FileNotFoundError("LHM++ priors are incomplete. Run its setup first.")
        if shutil.disk_usage(DOCKER_RESERVE).free < 15*1024**3: raise RuntimeError("LHM++ requires 15 GiB free at GSAVATAR_DOCKER_RESERVE_PATH.")
        lhmpp_image_info()
        flags=getattr(subprocess,"CREATE_NO_WINDOW",0)
        for kind, name in (("volume", LHMPP_VOLUME),):
            result=subprocess.run([DOCKER,kind,"inspect",name],capture_output=True,text=True,timeout=3,creationflags=flags)
            if result.returncode: raise RuntimeError("Start Docker Desktop and install the isolated LHM++ image/environment first.")
            info=json.loads(result.stdout)[0]
            labels=info.get("Labels",{}) if kind=="volume" else info.get("Config",{}).get("Labels",{})
            if (labels or {}).get("research.task")!="lhmpp": raise RuntimeError("Unexpected LHM++ environment label")
        _lhmpp_health=(time.monotonic(),True,None)
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as error:
        _lhmpp_health=(time.monotonic(),False,str(error))
    return _lhmpp_health[1:]


def model_choices():
    choices = []
    for model, spec in MODEL_SPECS.items():
        try:
            if not GENERATION_ENABLED:
                raise FileNotFoundError("Photo generation is disabled. See docs/generation.md to enable it.")
            if model != LHMPP_MODEL and (not GENERATION_PYTHON.is_file() or not LHM.is_dir()):
                raise FileNotFoundError("The separate LHM source / Python environment is missing.")
            checkpoint_path(model)
            available, reason = True, None
            if model==LHMPP_MODEL: available,reason=lhmpp_readiness()
        except (FileNotFoundError, ValueError) as error:
            available, reason = False, str(error)
        choices.append({"id": model, "label": spec["label"], "revision": spec["revision"],
                        "available": available, "reason": reason, "shapeModes":spec["shapeModes"],"maxPhotos":spec["maxPhotos"]})
    return choices


@contextmanager
def gpu_lease(path: Path | None = None, timeout: float = 0):
    """OS lock shared by HTTP jobs and lab runs; automatically released on exit/crash."""
    import os
    path = path or DATA / "generation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if stream.seek(0, 2) == 0:
        stream.write(b"0")
        stream.flush()
    deadline = time.monotonic() + timeout
    locked = False
    try:
        while not locked:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Another generation workload owns the GPU. Try again after it finishes.")
                time.sleep(0.25)
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
