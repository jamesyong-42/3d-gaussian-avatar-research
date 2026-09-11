"""Resolve saved generation inputs without exposing arbitrary filesystem paths."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


MAX_PHOTO_BYTES = 16 * 1024 * 1024


def matching_file(path: Path, root: Path, digest: str | None = None) -> Path | None:
    try:
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
            return None
        if not 0 < resolved.stat().st_size <= MAX_PHOTO_BYTES:
            return None
        if digest is not None:
            if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                return None
            if hashlib.sha256(resolved.read_bytes()).hexdigest() != digest.lower():
                return None
        return resolved
    except (OSError, RuntimeError):
        return None


@dataclass(frozen=True)
class SourcePhoto:
    index: int
    name: str
    path: Path | None

    def public(self, asset_id: str) -> dict:
        return {"index": self.index, "name": self.name, "available": self.path is not None,
                "url": f"/api/avatars/{asset_id}/source-photos/{self.index}" if self.path else None}


def source_photos(avatars: Path, asset_id: str, example: Path, bundled: Path):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", asset_id):
        raise ValueError("Invalid identifier")
    folder = (avatars / asset_id).resolve()
    if not folder.is_relative_to(avatars.resolve()):
        raise FileNotFoundError("Avatar not found")
    manifest = (folder / "avatar.json").resolve()
    if not manifest.is_relative_to(folder) or not manifest.is_file():
        raise FileNotFoundError("Avatar not ready")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Invalid avatar metadata")
    generation = meta.get("generation") or {}
    if not isinstance(generation, dict):
        raise ValueError("Invalid generation metadata")
    if not isinstance(generation.get("shape", {}), dict):
        raise ValueError("Invalid shape metadata")
    if generation.get("engine") == "synthetic-demo":
        return meta, []
    inputs = generation.get("inputs")
    if inputs is not None and (not isinstance(inputs, list) or not 1 <= len(inputs) <= 8
                               or not all(isinstance(item, dict) for item in inputs)):
        raise ValueError("Invalid input metadata")
    photos = []
    if inputs:
        # New uploads are normalized and numbered; old research exports retain
        # original bundled filenames/hashes instead of copying the input files.
        for index, item in enumerate(inputs):
            name = f"view-{index:03d}.png"
            digest = item.get("sha256")
            path = matching_file(folder / "inputs" / name, folder, digest)
            original = item.get("file", "")
            if path is None and isinstance(original, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,120}\.png", original) and digest:
                path = matching_file(bundled / original, bundled, digest)
            photos.append(SourcePhoto(index, name, path))
    elif meta.get("model") == "LHMPP-700M-PixelShuffle":
        count = generation.get("views", 1)
        if type(count) is not int or not 1 <= count <= 8:
            raise ValueError("Invalid input count")
        for index in range(count):
            name = f"view-{index:03d}.png"
            photos.append(SourcePhoto(index, name, matching_file(folder / "inputs" / name, folder)))
    else:
        path = matching_file(folder / "input.png", folder, generation.get("inputSha256"))
        # The original built-in avatar predates generation provenance metadata.
        # This is an explicit fixture mapping, never a metadata-supplied path.
        if path is None and asset_id == "example":
            path = matching_file(example, example.parent, generation.get("inputSha256"))
        photos.append(SourcePhoto(0, "input.png", path))
    return meta, photos
