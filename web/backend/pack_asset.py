"""Numeric-lossless sparse expression packaging; original v1 evidence is retained."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def write_arrays(folder, filename, arrays):
    specs, size, digest = {}, 0, hashlib.sha256()
    with (folder / filename).open("xb") as stream:
        for name, value in arrays.items():
            value = np.ascontiguousarray(value)
            pad = bytes((-size) % 4)
            stream.write(pad); digest.update(pad); size += len(pad)
            specs[name] = dict(offset=size, shape=list(value.shape), dtype=value.dtype.str, bytes=value.nbytes)
            payload = value.tobytes()
            stream.write(payload); digest.update(payload); size += len(payload)
    return dict(binary=filename, arrays=specs, bytes=size, sha256=digest.hexdigest())


def pack_asset(source, output, manifest_name="avatar.json"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if manifest_name not in ("avatar.json","avatar.pending.json"): raise ValueError("Unsupported manifest name")
    if source == output or any((output / n).exists() for n in ("avatar.json", "avatar.pending.json", "avatar.bin", "validation.bin")):
        raise ValueError("Choose a new output; original and existing assets are never replaced.")
    meta = json.loads((source / "avatar.json").read_text(encoding="utf-8"))
    if meta["version"] != 1 or meta["deformation"] != "lhm-linear-blend-v1":
        raise ValueError("Expected an original dense v1 rigged asset")
    binary = (source / "avatar.bin").read_bytes()
    if len(binary) != meta["bytes"] or hashlib.sha256(binary).hexdigest() != meta["sha256"]:
        raise ValueError("Source checksum mismatch")
    arrays = {}
    for name, spec in meta["arrays"].items():
        dtype = np.dtype(spec["dtype"])
        if spec["bytes"] != int(np.prod(spec["shape"])) * dtype.itemsize or spec["offset"] + spec["bytes"] > len(binary):
            raise ValueError(f"Invalid source array: {name}")
        arrays[name] = np.frombuffer(binary, dtype=dtype, count=int(np.prod(spec["shape"])), offset=spec["offset"]).reshape(spec["shape"])
    directions = arrays["expressionDirections"]
    if directions.shape != (meta["nExpressions"], meta["nGaussians"], 3) or not np.isfinite(directions).all():
        raise ValueError("Invalid expression basis")
    indices = np.flatnonzero(np.any(directions != 0, axis=(0, 2))).astype("<u4")
    sparse = np.ascontiguousarray(directions[:, indices, :])
    # Dropped rows are numerically zero (signed zero has identical deformation).
    if np.count_nonzero(sparse) != np.count_nonzero(directions):
        raise ValueError("Sparse conversion lost nonzero values")
    runtime = {k: v for k, v in arrays.items() if not k.startswith("reference_") and k != "expressionDirections"}
    runtime.update(expressionIndices=indices, expressionDirections=sparse)
    validation = {k: v for k, v in arrays.items() if k.startswith("reference_")}
    if len(meta.get("referencePoses", [])) != 7 or len(validation) != 14:
        raise ValueError("Seven complete native validation fixtures are required")
    output.mkdir(parents=True, exist_ok=True)
    packed = copy.deepcopy(meta)
    packed.update(version=2, expressionStorage="sparse-vertices-v1", nExpressionVertices=len(indices),
                  packaging=dict(sourceSha256=meta["sha256"], sourceBytes=len(binary),
                                 nonzeroExpressionValuesRetained=int(np.count_nonzero(sparse)),
                                 skinning="unchanged-all-nonzero-float32", validation="separate-on-demand"))
    packed.update(write_arrays(output, "avatar.bin", runtime))
    packed["validation"] = write_arrays(output, "validation.bin", validation)
    (output / manifest_name).write_text(json.dumps(packed, indent=2, allow_nan=False), encoding="utf-8")
    return packed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = pack_asset(args.source, args.output)
    print(json.dumps({k: meta[k] for k in ("version", "bytes", "sha256", "nExpressionVertices", "packaging")}))
