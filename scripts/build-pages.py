"""Build an explicit static-site allowlist; never package the application/data."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
output = ROOT / "_site"
output.mkdir(exist_ok=True)
if output.is_symlink() or output.resolve() != ROOT.resolve() / "_site":
    raise SystemExit("Unexpected site output location")
allowed = {"index.html", ".nojekyll"}
if any(path.name not in allowed or path.is_symlink() or not path.is_file() for path in output.iterdir()):
    raise SystemExit("Site output contains unexpected files; inspect it before building")
shutil.copyfile(ROOT / "docs/index.html", output / "index.html")
(output / ".nojekyll").touch()
print("Pages artifact: index.html + .nojekyll only. No API, photos, weights or avatar binaries.")
