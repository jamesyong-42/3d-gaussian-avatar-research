"""Prepare a bundled smoke input with IDOL's U2Net alpha-matting convention."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

LAB = Path(__file__).resolve().parents[1]
os.environ["U2NET_HOME"] = str(LAB / "checkpoints/IDOL/u2net")


def main():
    from PIL import Image, ImageOps
    from rembg import new_session, remove
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--shape-json', type=Path, help='Optional supplied SMPL-X reference (not an automatic estimate)')
    args = parser.parse_args()
    source = args.image.resolve()
    reference = None
    if args.shape_json:
        reference = json.loads(args.shape_json.read_text(encoding='utf-8'))
        raw = reference.get('shapes', reference.get('betas_save'))
        if raw and isinstance(raw[0], list): raw = raw[0]
        betas = [float(v) for v in raw[:10]]
        if len(betas) != 10 or not all(math.isfinite(v) for v in betas): raise ValueError('Need 10 finite supplied shape coefficients')
    output = LAB / "reports/idol" / ("input-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    started = time.perf_counter()
    session = new_session("u2net", providers=["CPUExecutionProvider"])
    with Image.open(source) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        rgb.save(output / "source.png")
        alpha = remove(rgb.convert("RGBA"), session=session, alpha_matting=True)
        alpha.save(output / "input.png")
    weights = Path(os.environ["U2NET_HOME"]) / "u2net.onnx"
    info = dict(source=str(source), sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                inputSha256=hashlib.sha256((output / "input.png").read_bytes()).hexdigest(),
                output=str(output), seconds=time.perf_counter()-started, mode=alpha.mode, size=list(alpha.size),
                preprocessing="rembg U2Net / alpha_matting=True on CPU; native 0.85 framing and 896x640 crop happen in the container",
                weightsSha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                scope="Bundled upstream image; engineering smoke only, no independent consent/quality audit")
    info['shapeMode'] = 'supplied-author-reference' if reference is not None else 'zero'
    info['betas'] = betas if reference is not None else [0.0] * 10
    if reference is not None:
        info.update(referenceSmplx=reference, referenceSha256=hashlib.sha256(args.shape_json.read_bytes()).hexdigest())
    (output / "input.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__": main()
