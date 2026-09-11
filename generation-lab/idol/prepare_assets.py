"""Pinned public IDOL downloads and inspected cache extraction; no model execution."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickletools
import shutil
import sys
import zipfile

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "lhmpp"))
from prepare_assets import fetch, digest  # Reuse the verified, resumable downloader.

REVISION = "3c80ec956f9bede01d9fbb3917ca229f8d2673b1"
SOURCE = "9fd9296c28e8f8f9ed5f5c594f3df1574b8ec82d"
DEST = LAB / "checkpoints/IDOL"
REPORT = LAB / "reports/idol-assets.json"
PINS = [
    ("cache_sub2.zip", 270183533, "fe833dc7c58b35d3b08a8ca8a82704d1c84aa9287afd8f5c47cafbdab01e718b"),
    ("model.ckpt", 8832048103, "2294c7714afd8eb883fcb8d9d386fb0573f066b0f3df9c44eacdea5ef440f180"),
    ("sapiens_1b_epoch_173_torchscript.pt2", 4678593669, "b5094b60a11ea8f94ef77b9e0a98439f677a5451c995e4dfda1de29538e8d861"),
]


def extract_cache():
    entries = []
    with zipfile.ZipFile(DEST / "cache_sub2.zip") as archive:
        if sum(i.file_size for i in archive.infolist()) > 4 * 1024**3:
            raise ValueError("Unexpected cache expansion size")
        for item in archive.infolist():
            raw_name = item.orig_filename  # Windows ZipInfo normalizes backslashes in .filename.
            name = PurePosixPath(raw_name)
            if name.is_absolute() or ".." in name.parts or "\\" in raw_name or ":" in raw_name or '\0' in raw_name or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive path")
            if item.is_dir(): continue
            allowed_cache = name.parts and name.parts[0] == "cache_sub2" and (name.suffix == ".npy" or str(name) == "cache_sub2/template/dense_thuman_newNeutral.obj")
            allowed_demo = name.parts and name.parts[0] == "demo_data" and name.suffix in (".jpg", ".json", ".npy")
            if not (allowed_cache or allowed_demo):
                raise ValueError("Unexpected cache member: " + str(name))
            target = DEST.joinpath(*name.parts)
            if not target.resolve().is_relative_to(DEST.resolve()): raise ValueError("Cache path escaped checkpoint folder")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as src, target.open("xb") as out: shutil.copyfileobj(src, out)
            if target.stat().st_size != item.file_size: raise ValueError("Partial cache member: " + str(target))
            # Existing files must match the verified archive, not just its lengths.
            expected = hashlib.sha256()
            with archive.open(item) as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''): expected.update(block)
            actual = digest(target)
            if actual != expected.hexdigest(): raise ValueError('Cache content mismatch: ' + str(target))
            entries.append(dict(name=str(name), bytes=item.file_size, sha256=actual))
    return entries


def inspect_serialization(filename):
    """Parse archive metadata and pickle opcodes only; never unpickle on the host."""
    with zipfile.ZipFile(DEST / filename) as archive:
        names = archive.namelist()
        globals_found = set()
        for name in names:
            if name.endswith(("data.pkl", "constants.pkl")):
                info = archive.getinfo(name)
                if info.file_size > 32 * 1024**2: raise ValueError("Unexpected pickle metadata size")
                for op, value, _ in pickletools.genops(archive.read(name)):
                    if op.name in ("GLOBAL", "INST"): globals_found.add(str(value))
        return dict(file=filename, members=len(names), pickleGlobals=sorted(globals_found),
                    torchscriptCodeFiles=sum("/code/" in name for name in names),
                    note="Static inspection only, not a security certification. Execute only in the network-disabled scoped container.")


def main():
    if shutil.disk_usage(LAB).free < 35 * 1024**3: raise RuntimeError("Need 35 GiB free on D: for IDOL weights, scratch, and evidence")
    DEST.mkdir(parents=True, exist_ok=True)
    report = dict(sourceRevision=SOURCE, modelRevision=REVISION, checkedAt=datetime.now(timezone.utc).isoformat(),
                  scope="Public author-hosted files, local research only. No credentials or new gated terms. Not product-use clearance.",
                  files=[dict(name=name, bytes=size, sha256=sha, target=str(DEST / name),
                              url=f"https://huggingface.co/yiyuzhuang/IDOL/resolve/{REVISION}/{name}") for name, size, sha in PINS])
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool: report["files"] = list(pool.map(fetch, report["files"]))
        report["cache"] = extract_cache()
        report["serialization"] = [inspect_serialization(name) for name in ("model.ckpt", "sapiens_1b_epoch_173_torchscript.pt2")]
        report["verified"] = True
    finally:
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"verified": True, "cacheFiles": len(report["cache"]), "report": str(REPORT)}))


if __name__ == "__main__": main()
