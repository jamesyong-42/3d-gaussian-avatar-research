"""Shared launch helpers; never select the separate CUDA environment implicitly."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def command(name):
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"Missing {name}. Install Python 3.10+ and Node.js 24 LTS, then reopen your terminal.")
    return found


def run(args, **kwargs):
    return subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True, **kwargs)


def in_venv():
    if not PYTHON.is_file():
        raise SystemExit("Run python scripts/setup.py first.")
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise SystemExit(subprocess.call([str(PYTHON), *sys.argv], cwd=ROOT))
    sys.path.insert(0, str(WEB / "backend"))


def demo():
    from settings import DATA
    run([command("node"), ROOT / "scripts/create-demo.mjs", DATA])
