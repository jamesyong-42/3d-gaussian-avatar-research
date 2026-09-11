"""Bounded IDOL compatibility trial; preserve all existing model environments."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

EXPERIMENT = Path(__file__).resolve().parent
LAB = EXPERIMENT.parent
sys.path.insert(0, str(LAB.parent / "web/backend"))
from generation_config import DOCKER, DOCKER_RESERVE, gpu_lease, lhmpp_image_info

SOURCE = "9fd9296c28e8f8f9ed5f5c594f3df1574b8ec82d"
VOLUME = "gaussian-idol-py310-env"
SCRATCH = LAB / "scratch/idol"
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def bind(source, target, readonly=True):
    return ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]


def call(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, creationflags=FLAGS, timeout=20, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "check", "native", "portability", "deformation"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--image", type=Path)
    parser.add_argument('--native-run', action='append', default=[], help='Completed native run ID, read-only input to portability')
    args = parser.parse_args()
    if not 30 <= args.timeout <= 1800: raise ValueError("Use a bounded 30–1800 second trial")
    if args.action in ('portability', 'deformation') and not 1 <= len(args.native_run) <= 2:
        raise ValueError('Deformation/portability requires one or two --native-run IDs')
    for run in args.native_run:
        if not re.fullmatch(r'native-\d{8}-\d{6}-[0-9a-f]{6}', run): raise ValueError('Invalid native run ID')
        record_path = LAB / 'reports/idol' / run / 'metrics.json'
        if not record_path.resolve().is_relative_to((LAB / 'reports/idol').resolve()): raise ValueError('Native run escaped reports')
        if json.loads(record_path.read_text(encoding='utf-8'))['status'] != 'native-passed': raise ValueError('Source native run did not pass')
    vendor = LAB / "vendors/IDOL"
    if call(["git", "rev-parse", "HEAD"], cwd=vendor).stdout.strip() != SOURCE or call(["git", "status", "--porcelain"], cwd=vendor).stdout.strip():
        raise RuntimeError("IDOL vendor source must match the clean pinned checkout")
    if shutil.disk_usage(DOCKER_RESERVE).free < 18 * 1024**3:
        raise RuntimeError("Need at least 18 GiB free at GSAVATAR_DOCKER_RESERVE_PATH before the compatibility trial; 15 GiB is the hard reserve")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    source = SCRATCH / "build-source"
    if not source.exists():
        shutil.copytree(vendor, source, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    tracked = call(['git', 'ls-files', '-z'], cwd=vendor, check=True).stdout.split('\0')
    for relative in filter(None, tracked):
        original, copied = vendor / relative, source / relative
        if not copied.is_file() or hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(copied.read_bytes()).digest():
            raise RuntimeError('Pinned IDOL code copy changed: ' + relative)
    # These are directories only in our disposable code copy, never the vendor.
    (source / "work_dirs").mkdir(exist_ok=True)
    (source / "lib/models/deformers/smplx/SMPLX").mkdir(exist_ok=True)
    image_info = lhmpp_image_info()
    result = call([DOCKER, "volume", "inspect", VOLUME])
    if result.returncode:
        if args.action != "install": raise RuntimeError("Install the isolated IDOL environment first")
        call([DOCKER, "volume", "create", "--label", "research.task=idol", VOLUME], check=True)
    elif json.loads(result.stdout)[0].get("Labels", {}).get("research.task") != "idol":
        raise RuntimeError("Unexpected volume ownership label")
    run_id = args.action + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    output = LAB / "reports/idol" / run_id
    output.mkdir(parents=True)
    name = "idol-" + run_id
    command = [DOCKER, "run", "--rm", "--name", name, "--label", "research.task=idol", "--label", "research.run=" + run_id,
               "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--cpus", "8", "--memory", "40g", "--shm-size", "4g",
               "--workdir", "/opt/idol", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "TORCH_CUDA_ARCH_LIST=8.9", "-e", "MAX_JOBS=4",
               "-e", "OMP_NUM_THREADS=6", "-e", "MKL_NUM_THREADS=6", "-e", "TMPDIR=/scratch/tmp", "-e", "TORCH_HOME=/scratch/torch"]
    command += ["--mount", f"type=volume,source={VOLUME},target=/opt/idol-env" + (",readonly" if args.action != "install" else "")]
    command += bind(SCRATCH, "/scratch", False) + bind(source, "/opt/idol") + bind(EXPERIMENT, "/experiment") + bind(output, "/evidence", False)
    if args.action != "install":
        command += ["--gpus", "all", "--network", "none", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"]
        command += bind(LAB / "checkpoints/IDOL", "/opt/idol/work_dirs")
        command += bind(LAB / "reports/idol-assets.json", "/asset-manifest.json")
        command += bind(LAB / "reports/lhmpp-assets.json", "/template-manifest.json")
        command += bind(LAB / "checkpoints/LHMPP-Prior/human_model_files/smplx", "/opt/idol/lib/models/deformers/smplx/SMPLX")
        for i, run in enumerate(args.native_run): command += bind(LAB / 'reports/idol' / run, f'/native/{i}')
        if args.image:
            args.image = args.image.resolve()
            if not args.image.is_file(): raise ValueError("Image not found")
            command += bind(args.image, "/input-photo")
            if (args.image.parent / "input.json").is_file(): command += bind(args.image.parent / "input.json", "/input-metadata.json")
    command += [image_info["Id"]]
    if args.action == 'install': command += ['bash', '/experiment/install_environment.sh']
    elif args.action == 'portability': command += ['/opt/idol-env/bin/python', '/experiment/portability_trial.py', '--sources', str(len(args.native_run))]
    elif args.action == 'deformation': command += ['/opt/idol-env/bin/python', '/experiment/deformation_trial.py', '--sources', str(len(args.native_run))]
    else: command += ['/opt/idol-env/bin/python', '/experiment/native_trial.py', '--stage', args.action]
    if args.action == "native" and args.image: command += ["--image", "/input-photo"]
    record = dict(id=run_id, sourceRevision=SOURCE, imageId=image_info["Id"], containerName=name, environmentVolume=VOLUME,
                  minimumSystemDriveFreeBytes=shutil.disk_usage(DOCKER_RESERVE).free,
                  startedAt=datetime.now(timezone.utc).isoformat(), command=command,
                  compatibility="PyTorch 2.3.0 / CUDA 12.1 / PyTorch3D 0.7.6, versus author's 2.3.1 / 11.8 / 0.7.7; no driver change",
                  input=str(args.image) if args.image else None, nativeRuns=args.native_run,
                  adapterHashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in EXPERIMENT.iterdir() if p.suffix in ('.py', '.sh', '.txt')})
    def save(): (output / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    save()
    print("START " + str(output), flush=True)
    with nullcontext() if args.action == "install" else gpu_lease(timeout=0):
        started = time.monotonic()
        with (output / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)
            try:
                while process.poll() is None:
                    try: process.wait(timeout=3)
                    except subprocess.TimeoutExpired: pass
                    if time.monotonic() - started > args.timeout: raise TimeoutError("Trial timeout")
                    free = shutil.disk_usage(DOCKER_RESERVE).free
                    record['minimumSystemDriveFreeBytes'] = min(record['minimumSystemDriveFreeBytes'], free)
                    if free < 15 * 1024**3: raise RuntimeError("Configured storage reserve reached; stopping only this trial")
            except BaseException as error:
                call([DOCKER, "stop", "--time", "5", name])
                process.wait(timeout=20)
                record["error"] = str(error)
            finally:
                record.update(exitCode=process.returncode, wallSeconds=time.monotonic()-started, finishedAt=datetime.now(timezone.utc).isoformat())
                save()
    print((output / "console.log").read_text(encoding="utf-8", errors="replace")[-9000:])
    print(json.dumps({k: record.get(k) for k in ("id", "exitCode", "wallSeconds", "error")}))
    return process.returncode or (1 if record.get("error") else 0)


if __name__ == "__main__": raise SystemExit(main())
