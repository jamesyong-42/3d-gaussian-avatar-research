"""Read-only configuration report; --create-demo explicitly adds synthetic data."""
import argparse
import json
import sys
from _common import in_venv, demo, WEB

in_venv()
from settings import DATA, GENERATION_ENABLED, GENERATION_PYTHON, LHM
from generation_config import model_choices

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--create-demo", action="store_true")
args = parser.parse_args()
if args.create_demo:
    demo()
print(json.dumps({"apiPython": sys.executable, "frontendBuilt": (WEB / "dist/index.html").is_file(),
    "dataDirectory": str(DATA), "generationEnabled": GENERATION_ENABLED,
    "generationPythonExists": GENERATION_PYTHON.is_file(), "lhmSourceExists": LHM.is_dir(),
    "models": model_choices()}, indent=2))
print("This checks files/configuration, not GPU inference or licenses. Do not publish reports containing local paths.")
