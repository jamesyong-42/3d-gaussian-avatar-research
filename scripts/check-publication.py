"""Audit the Git source set without printing possible secret values."""
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_PARTS = {"local", "third_party", "vendors", "environments", "checkpoints", "pretrained_models", "data", "scratch", "runs", "reports", "artifacts", "logs", "node_modules", "dist", ".venv", "bin", "obj", "test-results", "_site", ".git"}
FORBIDDEN_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mov", ".wav", ".fbx", ".ply", ".splat", ".ksplat", ".spz", ".bin", ".pkl", ".pt", ".pt2", ".pth", ".ckpt", ".safetensors", ".onnx", ".task", ".npy", ".npz", ".dll", ".exe", ".so", ".zip", ".tar", ".gz", ".pdf", ".pem", ".key"}
PATTERNS = {
    "private workstation path": re.compile(r"[A-Za-z]:[\\/](?:Users|I3T|CUDA)[\\/]|/(?:Users|home)/[a-zA-Z0-9_.-]+/"),
    "embedded raster image": re.compile(r"data:image/(?:png|jpe?g|webp|gif);base64,[A-Za-z0-9+/=]{1024,}", re.I),
    "GitHub credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "cloud access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private Tailscale identity": re.compile(r"[\w-]+\.tail[\w-]+\.ts\.net", re.I),
}
SOURCE_PREFIX = "https://github.com/jamesyong-42/3d-gaussian-avatar-research/blob/main/"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])


def source_files():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True, check=True)
    return sorted(set(filter(None, result.stdout.decode("utf-8").split("\0"))))


def audit():
    errors = []
    names = source_files()
    total = 0
    for name in names:
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"{name}: not a regular source file")
            continue
        if FORBIDDEN_PARTS.intersection(path.relative_to(ROOT).parts) or path.suffix.lower() in FORBIDDEN_EXT:
            errors.append(f"{name}: artifact or sensitive file type")
        if path.name.startswith(".env") and path.name != ".env.example":
            errors.append(f"{name}: local environment file")
        data = path.read_bytes()
        total += len(data)
        if len(data) > 1_000_000 or b"\0" in data:
            errors.append(f"{name}: oversized or binary content")
            continue
        try:
            content = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            errors.append(f"{name}: not UTF-8 source")
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{name}: {label} (value withheld)")
        links = []
        if path.suffix == ".md":
            links = re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", content)
        if name == "docs/index.html":
            page = Page()
            page.feed(content)
            if len(page.ids) != len(set(page.ids)):
                errors.append(f"{name}: duplicate HTML ids")
            links = page.links
            for target in links:
                if target.startswith("#") and target[1:] not in page.ids:
                    errors.append(f"{name}: broken section link {target}")
        for target in links:
            if target.startswith(SOURCE_PREFIX):
                local = ROOT / unquote(target[len(SOURCE_PREFIX):].split("#")[0])
            elif urlparse(target).scheme or target.startswith(("#", "//")):
                continue
            else:
                local = path.parent / unquote(target.split("#")[0])
            if not local.resolve().is_relative_to(ROOT) or not local.exists():
                errors.append(f"{name}: missing/out-of-repo linked source {target}")
    for error in errors:
        print(error)
    print(f"Publication audit: {len(names)} source files, {total:,} bytes, {len(errors)} findings. No ignored data is included.")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
