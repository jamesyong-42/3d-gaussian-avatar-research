"""Scoped Docker launcher for the isolated LHM++ experiment on this workstation."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import hashlib
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
from generation_config import DOCKER, DOCKER_RESERVE, MODEL_SPECS, LHMPP_MODEL, gpu_lease, lhmpp_image_info

IMAGE = "gaussian-lhmpp:cu121-source906b5d9"
ENV_VOLUME = "gaussian-lhmpp-py310-env"
SCRATCH = LAB / "scratch/lhmpp"
REPORTS = LAB / "reports/lhmpp"
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def mount(source, target, readonly=False):
    return ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "check", "native"))
    parser.add_argument("--views", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--wait-lock", type=float, default=0)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    if args.job_id and not re.fullmatch(r"[a-f0-9]{32}", args.job_id): raise ValueError("Invalid job identifier")
    if args.record and args.record.exists(): raise ValueError("Refusing to overwrite an existing run record")
    if args.input_dir:
        args.input_dir=args.input_dir.resolve()
        images=sorted(args.input_dir.glob("*.png"))
        if args.action!="native" or len(images)!=args.views or not 1<=len(images)<=8:
            raise ValueError("Input directory must contain exactly the requested 1–8 normalized PNG photos")
    for directory in (SCRATCH, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DOCKER_RESERVE).free < 15 * 1024**3:
        raise RuntimeError("Less than 15 GiB free at GSAVATAR_DOCKER_RESERVE_PATH. Stop before consuming more Docker storage.")
    inspected = subprocess.run([DOCKER, "volume", "inspect", ENV_VOLUME], capture_output=True, text=True, creationflags=FLAGS)
    if inspected.returncode:
        if args.action != "install":
            raise RuntimeError("Install the isolated Linux environment first.")
        subprocess.run([DOCKER, "volume", "create", "--label", "research.task=lhmpp", ENV_VOLUME], check=True, capture_output=True, creationflags=FLAGS)
    elif json.loads(inspected.stdout)[0].get("Labels", {}).get("research.task") != "lhmpp":
        raise RuntimeError("Existing volume is not labeled for this experiment; refusing to use it.")
    run_id = f"{args.action}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    output = REPORTS / run_id
    output.mkdir()
    container_name = "lhmpp-job-" + args.job_id if args.job_id else "lhmpp-" + run_id
    cmd = [DOCKER, "run", "--rm", "--name", container_name, "--label", "research.task=lhmpp",
           "--security-opt", "no-new-privileges", "--cpus", "8", "--memory", "36g", "--shm-size", "4g",
           "-e", "OMP_NUM_THREADS=6", "-e", "MKL_NUM_THREADS=6", "-e", "PYTHONDONTWRITEBYTECODE=1"]
    cmd += ["--mount", f"type=volume,source={ENV_VOLUME},target=/opt/lhmpp-env" + (",readonly" if args.action != "install" else "")]
    cmd += mount(SCRATCH, "/scratch") + mount(EXPERIMENT, "/experiment", True) + mount(output, "/evidence")
    if args.action != "install":
        cmd += ["--gpus", "all", "--network", "none", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"]
    if args.action == "native":
        cmd += mount(LAB / "checkpoints/LHMPP-Prior", "/opt/lhmpp/pretrained_models", True)
        cmd += mount(MODEL_SPECS[LHMPP_MODEL]["path"], "/checkpoint", True)
        cmd += mount(LAB.parent / "web/backend", "/portable-backend", True)
        if args.input_dir: cmd += mount(args.input_dir, "/inputs", True)
        if args.job_id: cmd += ["--label", "research.job=" + args.job_id]
    image_info = lhmpp_image_info()
    cmd += [image_info["Id"]]
    if args.action == "install":
        cmd += ["bash", "/experiment/install_environment.sh"]
    elif args.action == "check":
        cmd += ["/opt/lhmpp-env/bin/python", "/experiment/preflight.py", "/evidence/environment-check.json"]
    else:
        cmd += ["/opt/lhmpp-env/bin/python", "/experiment/native_trial.py", "--views", str(args.views)]
        if args.input_dir: cmd += ["--input-dir", "/inputs"]
    record = {"id": run_id, "action": args.action, "startedAt": datetime.now(timezone.utc).isoformat(),
              "command": cmd, "network": "install-only" if args.action == "install" else "disabled",
              "output": str(output), "views": args.views if args.action == "native" else None,
              "imageId": image_info["Id"], "sourceRevision": image_info["Config"]["Labels"].get("research.source"),
              "environmentVolume": ENV_VOLUME, "containerName": container_name,
              "inputDirectory": str(args.input_dir) if args.input_dir else None,
              "adapterHashes": {name: hashlib.sha256((EXPERIMENT/name).read_bytes()).hexdigest() for name in ("native_trial.py", "export_portable.py", "container.py")}}
    def save_record():
        (output / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if args.record: args.record.write_text(json.dumps(record, indent=2), encoding="utf-8")
    save_record()
    print(f"START {container_name}: {output}", flush=True)
    print('GSAVATAR '+json.dumps({"stage":"Waiting for the isolated GPU worker", "progress":.08}),flush=True)
    with (nullcontext() if args.action == "install" else gpu_lease(timeout=args.wait_lock)):
        started = time.monotonic()
        with (output / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)
            try:
                last_stage=""
                while process.poll() is None:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        tail=(output/"console.log").read_text(encoding="utf-8",errors="replace")[-5000:]
                        stage,progress="Loading LHM++ in isolated CUDA worker",.18
                        if "NATIVE reconstructing" in tail: stage,progress="Reconstructing from the uploaded views",.55
                        if "NATIVE exporting portable" in tail: stage,progress="Exporting rig and native validation fixtures",.72
                        if stage!=last_stage:
                            print('GSAVATAR '+json.dumps({"stage":stage,"progress":progress}),flush=True);last_stage=stage
                    if time.monotonic() - started > args.timeout:
                        raise TimeoutError(f"Exceeded {args.timeout}s")
                    if shutil.disk_usage(DOCKER_RESERVE).free < 15 * 1024**3:
                        raise RuntimeError("Configured storage reserve dropped below 15 GiB; stopping only this experiment container.")
            except BaseException as error:
                # Stop only our explicitly named container; never kill unrelated
                # Python processes or reset Docker/WSL. --rm removes its ephemeral
                # writable layer; all mounted experiment artifacts are retained.
                subprocess.run([DOCKER, "stop", "--time", "10", container_name], capture_output=True, creationflags=FLAGS)
                process.wait(timeout=30)
                record.update(exitCode=process.returncode, wallSeconds=round(time.monotonic()-started, 3),
                              error=f"{type(error).__name__}: {error}", finishedAt=datetime.now(timezone.utc).isoformat())
                save_record()
                raise
    record.update(exitCode=process.returncode, wallSeconds=round(time.monotonic() - started, 3),
                  finishedAt=datetime.now(timezone.utc).isoformat())
    save_record()
    print((output / "console.log").read_text(encoding="utf-8", errors="replace")[-7000:], flush=True)
    print(json.dumps({"exitCode": process.returncode, "wallSeconds": record["wallSeconds"], "report": str(output)}), flush=True)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
