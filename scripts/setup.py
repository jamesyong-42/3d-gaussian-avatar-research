"""Prepare the viewer, API and tests. Does NOT install CUDA or download models."""
import subprocess
import sys
from _common import ROOT, WEB, PYTHON, command, run


def main():
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")
    node = command("node")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if int(version.lstrip("v").split(".")[0]) < 24:
        raise SystemExit("Node.js 24+ required; Node.js 24 LTS is recommended.")
    npm = command("npm")
    if not PYTHON.is_file():
        run([sys.executable, "-m", "venv", ROOT / ".venv"])
    run([PYTHON, "-m", "pip", "install", "-r", WEB / "requirements-dev.txt"])
    run([npm, "--prefix", WEB, "ci"])
    run([npm, "--prefix", WEB, "run", "build"])
    run([PYTHON, ROOT / "scripts/doctor.py", "--create-demo"])
    print("\nReady. Run: python scripts/serve.py\nOpen: http://127.0.0.1:8765\nGeneration is opt-in; see docs/generation.md.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
