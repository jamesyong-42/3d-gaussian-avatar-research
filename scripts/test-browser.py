"""Run browser smoke tests against an owned process and disposable synthetic data."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from urllib.error import URLError
from _common import ROOT, WEB, in_venv, command, run

in_venv()
with tempfile.TemporaryDirectory(prefix="gaussian-browser-smoke-") as data:
    env = dict(os.environ, GSAVATAR_ENABLE_GENERATION="0", GSAVATAR_DATA_DIR=data)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env["AVATAR_LAB_URL"] = f"http://127.0.0.1:{port}"
    worker = subprocess.Popen([sys.executable, str(ROOT / "scripts/serve.py"), "--port", str(port)], cwd=ROOT, env=env)
    try:
        for attempt in range(120):
            if worker.poll() is not None:
                raise RuntimeError("Owned smoke-test server stopped during startup")
            try:
                with urlopen(env["AVATAR_LAB_URL"] + "/api/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                time.sleep(.25)
        else:
            raise TimeoutError("Smoke-test server did not become ready")
        run([command("npm"), "--prefix", WEB, "run", "test:browser"], env=env)
        run([command("node"), ROOT / "scripts/check-pages.mjs"], env=env)
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)
