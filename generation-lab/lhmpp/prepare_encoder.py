"""Reuse the already-present DINOv2 prior without modifying the user's cache."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_assets import digest, LAB

name = "dinov2_vitl14_reg4_pretrain.pth"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints" / name)
source = parser.parse_args().source.expanduser().resolve()
target = LAB / "scratch/lhmpp/torch/hub/checkpoints" / name
expected = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
if not source.is_file() or digest(source) != expected:
    raise ValueError("The inspected existing DINOv2 cache is missing or changed; do not silently substitute it.")
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists() and digest(target) != expected:
    raise ValueError("Refusing to overwrite a different cached encoder.")
if not target.exists():
    shutil.copy2(source, target)
record = {"file": name, "bytes": target.stat().st_size, "sha256": digest(target), "localSource": str(source),
          "officialUrl": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
          "provenance": "Copied the inspected pre-existing cache; hash pins local bytes, not a separate signature verification."}
(LAB / "reports").mkdir(parents=True, exist_ok=True)
(LAB / "reports/lhmpp-encoder.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
