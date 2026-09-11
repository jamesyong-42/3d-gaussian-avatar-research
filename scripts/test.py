"""Run asset-free tests and the production frontend build; no GPU/downloads."""
import subprocess
import sys
from _common import in_venv, ROOT, WEB, command, run

in_venv()
try:
    run([command("npm"), "--prefix", WEB, "test"])
    for directory in (WEB / "tests", ROOT / "generation-lab", ROOT / "generation-lab/idol"):
        run([sys.executable, "-m", "unittest", "discover", "-s", directory, "-p", "test_*.py"])
    run([command("node"), "--test", "--test-isolation=none", ROOT / "generation-lab/idol/deformation_cpu.test.mjs"])
    run([command("npm"), "--prefix", WEB, "run", "build"])
except subprocess.CalledProcessError as error:
    raise SystemExit(error.returncode) from error
