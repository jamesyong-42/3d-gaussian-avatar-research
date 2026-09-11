"""One bounded HTTP/CLI job: isolated native inference -> compact validated asset."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from generation_config import LAB, lhmpp_readiness
from pack_asset import pack_asset


def event(stage, progress):
    print("GSAVATAR " + json.dumps(dict(stage=stage, progress=progress)), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--job-id",required=True)
    args=parser.parse_args()
    ready,reason=lhmpp_readiness(refresh=True)
    if not ready: raise RuntimeError(reason)
    images=sorted(args.input_dir.glob("*.png"))
    if not 1<=len(images)<=8: raise ValueError("LHM++ needs 1–8 normalized photos")
    sys.path.insert(0,str(LAB/"lhmpp"))
    import container
    record=args.output/"lhmpp-run.json"
    code=container.main(["native","--views",str(len(images)),"--timeout","360","--wait-lock","120",
                         "--input-dir",str(args.input_dir),"--job-id",args.job_id,"--record",str(record)])
    if code: raise RuntimeError(f"Native LHM++ container exited with code {code}; see generation.log")
    run=json.loads(record.read_text(encoding="utf-8"));native=Path(run["output"])
    metrics=json.loads((native/"metrics.json").read_text(encoding="utf-8"))
    if metrics["status"]!="native-passed" or not metrics.get("checkpointCoverageVerified") or metrics.get("portable")!="exported-awaiting-independent-validation":
        raise RuntimeError("Native LHM++ reconstruction/export gate did not pass")
    event("Packing sparse expressions and separating validation data",.86)
    meta=pack_asset(native,args.output,manifest_name="avatar.pending.json")
    # Keep the library entry hidden until the independent validation gate passes.
    pending=args.output/"avatar.pending.json"
    try:
        event("Checking all seven native poses independently",.94)
        check=subprocess.run(["node",str(LAB/"validate_asset.mjs"),str(args.output),"--pending"],capture_output=True,text=True,timeout=90,
                             creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if check.returncode: raise RuntimeError("Portable validation failed: "+check.stdout+check.stderr)
        verified=json.loads(check.stdout)
        (args.output/"validation-results.json").write_text(json.dumps(verified,indent=2),encoding="utf-8")
        meta["label"]=f"Your photos / LHM++ / {len(images)} view{'s' if len(images)!=1 else ''}"
        metrics.update(runtimeBytes=meta["bytes"],validationBytes=meta["validation"]["bytes"],portable="validated-compact-v2",packagerSha256=hashlib.sha256(Path(__file__).with_name("pack_asset.py").read_bytes()).hexdigest())
        (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        pending.write_text(json.dumps(meta,indent=2),encoding="utf-8")
        pending.rename(args.output/"avatar.json")
    except Exception:
        # Failed evidence is retained but no avatar.json publishes it to clients.
        raise
    event("LHM++ avatar validated; GPU worker has exited",1)


if __name__=="__main__": main()
