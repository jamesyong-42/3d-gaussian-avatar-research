"""Validate and copy a native trial into a new web-library entry; never replace one."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image
from prepare_assets import LAB, digest

WEB = LAB.parent / "web"
sys.path.insert(0, str(WEB / "backend"))
from settings import DATA


def publish(folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to((LAB / "reports/lhmpp").resolve()) or not folder.name.startswith("native-"):
        raise ValueError("Expected an LHM++ native trial directory")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if metrics["status"] != "native-passed" or metrics.get("portable") != "exported-awaiting-independent-validation" or not metrics.get("checkpointCoverageVerified"):
        raise ValueError("Native trial has not passed strict reconstruction/export gates")
    check = subprocess.run(["node", str(LAB / "validate_asset.mjs"), str(folder)], capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if check.returncode:
        raise ValueError(check.stdout + check.stderr)
    validated = json.loads(check.stdout)
    (folder / "node-validation.json").write_text(json.dumps(validated, indent=2), encoding="utf-8")
    asset_id = f"lhmpp-{metrics['views']}v-{folder.name.removeprefix('native-')}"
    target = DATA / "avatars" / asset_id
    if target.exists():
        raise ValueError("Library entry already exists; refusing to overwrite it")
    target.mkdir()
    for name in ("avatar.bin", "metrics.json"):
        shutil.copy2(folder / name, target / name)
    meta = json.loads((folder / "avatar.json").read_text(encoding="utf-8"))
    meta["label"] = f"LHM++ / {metrics['views']} view{'s' if metrics['views'] != 1 else ''} / bundled example"
    (target / "avatar.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    source = LAB / "vendors/LHM-plusplus/assets/example_multi_images" / metrics["inputs"][0]["file"]
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((256, 256))
        image.save(target / "thumbnail.jpg", quality=85)
    if digest(target / "avatar.bin") != meta["sha256"]:
        raise ValueError("Published copy failed checksum")
    result = {"assetId": asset_id, "source": str(folder), "views": metrics["views"], "validation": validated,
              "url": f"http://127.0.0.1:8765/?asset={asset_id}"}
    (folder / "published.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    publish(sys.argv[1])
