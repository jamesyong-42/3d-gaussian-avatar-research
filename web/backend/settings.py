"""Portable, local-only runtime configuration. No model imports or installation."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
WEB = ROOT / "web"
try:
    from dotenv import load_dotenv
except ImportError:
    # Native exporter processes inherit the launcher environment. The lightweight
    # API installs python-dotenv; separate CUDA environments need not import it.
    pass
else:
    load_dotenv(ROOT / ".env", override=False)


def configured_path(name: str, default: Path | str) -> Path:
    value = Path(os.environ.get(name) or default).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


DATA = configured_path("GSAVATAR_DATA_DIR", ROOT / "local/data")
LHM = configured_path("GSAVATAR_LHM_ROOT", ROOT / "third_party/LHM")
LAB = ROOT / "generation-lab"
_default_python = LHM / ("lhm_env/Scripts/python.exe" if os.name == "nt" else "lhm_env/bin/python")
GENERATION_PYTHON = configured_path("GSAVATAR_GENERATION_PYTHON", _default_python)
GENERATION_ENABLED = os.environ.get("GSAVATAR_ENABLE_GENERATION", "0") == "1"
DOCKER = os.environ.get("GSAVATAR_DOCKER") or shutil.which("docker") or "docker"
DOCKER_RESERVE = configured_path("GSAVATAR_DOCKER_RESERVE_PATH", Path(ROOT.anchor))
EXAMPLE_PHOTO = configured_path("GSAVATAR_EXAMPLE_PHOTO", ROOT / "local/consented-example.png")


def generation_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cuda = env.get("GSAVATAR_CUDA_HOME")
    if cuda:
        toolkit = configured_path("GSAVATAR_CUDA_HOME", cuda)
        env.update(CUDA_HOME=str(toolkit), CUDA_PATH=str(toolkit))
        env["PATH"] = str(toolkit / "bin") + os.pathsep + env.get("PATH", "")
    return env
