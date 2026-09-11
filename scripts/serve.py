"""Serve one loopback-only API/UI process. Use an authenticated gateway to share."""
import argparse
from _common import in_venv, WEB, demo

in_venv()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()
if not 1024 <= args.port <= 65535:
    parser.error("port must be 1024–65535")
if not (WEB / "dist/index.html").is_file():
    raise SystemExit("Frontend build is missing. Run python scripts/setup.py.")
demo()
import uvicorn

print(f"Local viewer: http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
uvicorn.run("app:app", host="127.0.0.1", port=args.port, workers=1, access_log=False)
