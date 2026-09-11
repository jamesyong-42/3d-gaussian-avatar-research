"""Local upload/job/asset API and pose relay. No neural rendering in this server."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from generation_config import DEFAULT_MODEL, DEFAULT_SHAPE, MODEL_SPECS, LHMPP_MODEL, DOCKER, checkpoint_path, model_choices, validate_choice, lhmpp_readiness
from source_photos import source_photos
from settings import DATA, LHM, GENERATION_PYTHON, GENERATION_ENABLED, EXAMPLE_PHOTO, generation_environment

WEB = Path(__file__).resolve().parents[1]
AVATARS = DATA / "avatars"
JOBS = DATA / "jobs"
PYTHON = GENERATION_PYTHON
EXAMPLE = EXAMPLE_PHOTO
BUNDLED_PHOTOS = WEB.parent / "generation-lab" / "vendors" / "LHM-plusplus" / "assets" / "example_multi_images"
PHOTO_HEADERS = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}
MAX_UPLOAD = 16 * 1024 * 1024
MAX_TOTAL_UPLOAD = 64 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 24_000_000
jobs: dict[str, dict] = {}
job_lock = threading.Lock()
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avatar-generation")
running: subprocess.Popen | None = None
running_job_id: str | None = None
rooms: dict[str, dict] = {}


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise HTTPException(400, "Invalid identifier")
    return value


def persist_job(job_id, **updates):
    with job_lock:
        jobs[job_id].update(updates)
        payload = dict(jobs[job_id])
        (JOBS / f"{job_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def run_job(job_id, source, output, model=DEFAULT_MODEL, shape_mode=DEFAULT_SHAPE):
    global running, running_job_id
    running_job_id=job_id
    started = time.monotonic()
    persist_job(job_id, status="running", stage="Starting GPU worker", progress=0.02)
    env = generation_environment()
    try:
        if model==LHMPP_MODEL:
            command=[sys.executable,"-u",str(WEB/"backend/run_lhmpp.py"),"--input-dir",str(source),"--output",str(output),"--job-id",job_id]
        else:
            command=[str(PYTHON),"-u",str(WEB/"backend/export_lhm.py"),"--image",str(source),"--output",str(output),"--model",model,"--shape-mode",shape_mode,"--wait-lock","120"]
        with (output / "generation.log").open("w", encoding="utf-8") as log:
            running = subprocess.Popen(command, cwd=WEB, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            tail=[]
            for line in running.stdout:
                log.write(line); log.flush()
                tail.append(line.strip()); tail=tail[-15:]
                if line.startswith("GSAVATAR "):
                    event=json.loads(line[len("GSAVATAR "):])
                    persist_job(job_id, stage=event["stage"], progress=event["progress"])
            code=running.wait()
            if code or not (output / "avatar.json").is_file():
                raise RuntimeError("Reconstruction failed. " + " ".join(tail[-5:]))
        persist_job(job_id,status="complete",stage="Ready",progress=1,seconds=round(time.monotonic()-started,2))
    except Exception as error:
        persist_job(job_id,status="failed",stage="Generation failed",error=str(error),seconds=round(time.monotonic()-started,2))
    finally:
        running=None
        running_job_id=None


def stop_owned_container(job_id):
    """Shutdown/recovery only for a container carrying this exact job's label."""
    if not re.fullmatch(r"[a-f0-9]{32}",job_id): return
    name="lhmpp-job-"+job_id
    try:
        flags=getattr(subprocess,"CREATE_NO_WINDOW",0)
        inspected=subprocess.run([DOCKER,"container","inspect",name],capture_output=True,text=True,timeout=3,creationflags=flags)
        if inspected.returncode: return
        labels=json.loads(inspected.stdout)[0]["Config"].get("Labels",{}) or {}
        if labels.get("research.job")==job_id and labels.get("research.task")=="lhmpp":
            subprocess.run([DOCKER,"stop","--time","5",name],capture_output=True,timeout=10,creationflags=flags)
    except (OSError,ValueError,KeyError,subprocess.TimeoutExpired): pass


@asynccontextmanager
async def lifespan(app):
    AVATARS.mkdir(parents=True,exist_ok=True); JOBS.mkdir(parents=True,exist_ok=True)
    for path in JOBS.glob("*.json"):
        try:
            job=json.loads(path.read_text(encoding="utf-8"))
            jobs[job["id"]]=job
            if job["status"] in ("queued","running"):
                if job.get("model")==LHMPP_MODEL: stop_owned_container(job["id"])
                persist_job(job["id"],status="failed",stage="Interrupted",error="The server stopped during generation. Submit the photo again.")
        except (ValueError,KeyError):
            pass
    yield
    if running and running.poll() is None:
        if running_job_id and jobs.get(running_job_id,{}).get("model")==LHMPP_MODEL: stop_owned_container(running_job_id)
        running.terminate()
    pool.shutdown(wait=False,cancel_futures=True)


app=FastAPI(title="Gaussian Avatar Lab",lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","testserver"])


@app.middleware("http")
async def local_origin(request: Request, call_next):
    from urllib.parse import urlparse
    if request.method not in ("GET","HEAD","OPTIONS"):
        origin=request.headers.get("origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1","localhost"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail":"This service only accepts local browser requests."},status_code=403)
        length=request.headers.get("content-length","")
        if request.url.path=="/api/jobs" and length.isdecimal() and int(length)>MAX_TOTAL_UPLOAD+1024*1024:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail":"Keep the combined upload below 64 MB."},status_code=413)
    return await call_next(request)


@app.get("/api/health")
def health():
    choices = model_choices()
    return {"ok":True,"model":DEFAULT_MODEL,"workerAvailable":any(item["available"] for item in choices),
            "generationEnabled":GENERATION_ENABLED,"examplePhotoAvailable":EXAMPLE.is_file(),"workerRunning":running is not None,"singlePhoto":True,
            "models":choices,"defaultShape":DEFAULT_SHAPE,"shapeModes":["zero","estimate"],"multiPhoto":True,"maxTotalUploadBytes":MAX_TOTAL_UPLOAD}


@app.get("/api/avatars")
def avatars():
    result=[]
    for path in AVATARS.glob("*/avatar.json"):
        try:
            meta=json.loads(path.read_text(encoding="utf-8"))
            label = meta.get("label", "Example · full body" if path.parent.name=="example" else "Photo avatar")
            result.append({"id":path.parent.name,"label":label,"model":meta.get("model",DEFAULT_MODEL),
                           "shapeMode":meta.get("generation",{}).get("shape",{}).get("mode","zero"),
                           "nGaussians":meta["nGaussians"],"createdAt":meta["createdAt"],"bytes":meta["bytes"]})
        except (ValueError,KeyError):
            pass
    return sorted(result,key=lambda item:item["createdAt"],reverse=True)


@app.get("/api/example-photo")
def example_photo():
    if not EXAMPLE.is_file(): raise HTTPException(404, "No consented example photo is configured. Upload your own photo.")
    return FileResponse(EXAMPLE,media_type="image/png",filename="example-full-body.png")


@app.post("/api/jobs",status_code=202)
async def create_job(photo: UploadFile | None=File(None), photos: list[UploadFile] | None=File(None), model: str=Form(DEFAULT_MODEL), shape_mode: str | None=Form(None)):
    if not GENERATION_ENABLED:
        raise HTTPException(409, "Photo generation is disabled. Complete docs/generation.md and explicitly enable it.")
    shape_mode=shape_mode or ("predicted" if model==LHMPP_MODEL else DEFAULT_SHAPE)
    try:
        validate_choice(model,shape_mode)
        checkpoint_path(model)
    except ValueError as error:
        raise HTTPException(400,str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(409,str(error)) from error
    if model==LHMPP_MODEL:
        available,reason=lhmpp_readiness(refresh=True)
        if not available: raise HTTPException(409,reason)
    if photo is not None and photos: raise HTTPException(400,"Use photo or photos, not both.")
    selected=photos or ([photo] if photo is not None else [])
    if not 1<=len(selected)<=MODEL_SPECS[model]["maxPhotos"]:
        raise HTTPException(400,f"{model} accepts 1–{MODEL_SPECS[model]['maxPhotos']} photos of the same person.")
    if sum(j["status"] in ("queued","running") for j in jobs.values()) >= 3:
        raise HTTPException(429,"The GPU queue is full. Wait for an existing job to finish.")
    images=[];total=0
    for item in selected:
        payload=await item.read(MAX_UPLOAD+1);total+=len(payload)
        if len(payload)>MAX_UPLOAD or total>MAX_TOTAL_UPLOAD:
            raise HTTPException(413,"Use photos under 16 MB each and 64 MB combined.")
        try:
            with Image.open(BytesIO(payload)) as uploaded:
                if uploaded.format not in ("JPEG","PNG","WEBP"): raise HTTPException(400,"Use JPG, PNG, or WebP photos.")
                if uploaded.width*uploaded.height>24_000_000 or min(uploaded.size)<128:
                    raise HTTPException(400,"Choose photos of at least 128 pixels per side and at most 24 megapixels.")
                image=ImageOps.exif_transpose(uploaded).convert("RGB");image.thumbnail((2048,2048));images.append(image)
        except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
            raise HTTPException(400,"Every upload must be a valid JPG, PNG, or WebP photo.") from error
    job_id=uuid.uuid4().hex
    with job_lock:
        # Recheck after asynchronous upload reads so parallel requests cannot
        # bypass admission limits. Invalid batches create no job/output folder.
        if sum(j["status"] in ("queued","running") for j in jobs.values())>=3: raise HTTPException(429,"The GPU queue is full.")
        output=AVATARS/job_id; output.mkdir(parents=True)
        if model==LHMPP_MODEL:
            source=output/"inputs";source.mkdir()
            for i,image in enumerate(images): image.save(source/f"view-{i:03d}.png")
        else:
            source=output/"input.png";images[0].save(source)
        preview=images[0].copy();preview.thumbnail((256,256));preview.save(output/"thumbnail.jpg",quality=85)
        jobs[job_id]={"id":job_id,"assetId":job_id,"status":"queued","stage":"Waiting for the GPU","progress":0,"createdAt":time.time(),"model":model,"shapeMode":shape_mode,"photoCount":len(images)}
    persist_job(job_id)
    pool.submit(run_job,job_id,source,output,model,shape_mode)
    return dict(jobs[job_id])


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    safe_id(job_id)
    if job_id not in jobs: raise HTTPException(404,"Job not found")
    return dict(jobs[job_id])


def avatar_source_photos(asset_id: str):
    safe_id(asset_id)
    try:
        return source_photos(AVATARS, asset_id, EXAMPLE, BUNDLED_PHOTOS)
    except FileNotFoundError as error:
        raise HTTPException(404, "Avatar not ready") from error
    except (ValueError, TypeError, AttributeError, OSError) as error:
        raise HTTPException(409, "Saved input metadata is unavailable") from error


@app.get("/api/avatars/{asset_id}/source-photos")
def source_photo_list(asset_id: str):
    meta, photos = avatar_source_photos(asset_id)
    return JSONResponse({"assetId": asset_id, "model": meta.get("model"),
                         "shapeMode": (meta.get("generation") or {}).get("shape", {}).get("mode", "zero"),
                         "synthetic": meta.get("generation", {}).get("engine") == "synthetic-demo",
                         "total": len(photos), "photos": [photo.public(asset_id) for photo in photos]},
                        headers=PHOTO_HEADERS)


@app.get("/api/avatars/{asset_id}/source-photos/{index}")
def source_photo_file(asset_id: str, index: int):
    if not 0 <= index < 8:
        raise HTTPException(404, "Source photo not found")
    _, photos = avatar_source_photos(asset_id)
    if index >= len(photos) or photos[index].path is None:
        raise HTTPException(404, "Source photo unavailable")
    return FileResponse(photos[index].path, media_type="image/png", headers=PHOTO_HEADERS)


@app.get("/api/avatars/{asset_id}/download")
def download(asset_id: str):
    folder=AVATARS/safe_id(asset_id)
    if not (folder/"avatar.json").is_file(): raise HTTPException(404,"Avatar not ready")
    archive=folder/"avatar.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for name in ("avatar.json","avatar.bin"):
                z.write(folder/name,name)
            if (folder/"validation.bin").is_file(): z.write(folder/"validation.bin","validation.bin")
    return FileResponse(archive,filename=f"avatar-{asset_id}.zip")


@app.get("/api/avatars/{asset_id}/{filename}")
def asset_file(asset_id: str,filename: str):
    # An extensionless inline route prevents desktop download managers from
    # swallowing browser fetches for .bin files (observed with IDM on this PC).
    if filename in ("data", "validation"):
        path=AVATARS/safe_id(asset_id)/("avatar.bin" if filename=="data" else "validation.bin")
        if not path.is_file(): raise HTTPException(404,"Avatar data not found")
        return FileResponse(path,media_type="application/vnd.gsavatar",headers={"Content-Disposition":"inline","X-Content-Type-Options":"nosniff"})
    if filename not in ("avatar.json","avatar.bin","thumbnail.jpg","generation.log","metrics.json","shape.json"): raise HTTPException(404)
    path=AVATARS/safe_id(asset_id)/filename
    if not path.is_file(): raise HTTPException(404,"Asset file not found")
    return FileResponse(path)


@app.websocket("/ws/{room_id}")
async def room_socket(ws: WebSocket,room_id: str):
    from urllib.parse import urlparse
    origin=ws.headers.get("origin")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}",room_id) or (origin and urlparse(origin).hostname not in ("127.0.0.1","localhost")):
        await ws.close(code=1008); return
    await ws.accept()
    room=rooms.setdefault(room_id,{"owner":None,"clients":set(),"asset":None})
    owner=ws.query_params.get("role")=="sender"
    if owner and room["owner"] is not None:
        await ws.send_json({"type":"error","message":"This room already has a sender."}); await ws.close(code=1008); return
    if owner: room["owner"]=ws
    room["clients"].add(ws)
    if room["asset"]: await ws.send_json({"type":"asset","id":room["asset"]})
    async def broadcast(message):
        for other in list(room["clients"]):
            if other is ws: continue
            try:
                if isinstance(message,bytes): await other.send_bytes(message)
                else: await other.send_json(message)
            except (RuntimeError,WebSocketDisconnect): room["clients"].discard(other)
    try:
        while True:
            message=await ws.receive()
            if message["type"]=="websocket.disconnect": break
            if not owner: continue
            blob=message.get("bytes")
            if blob is not None:
                if len(blob)>2048 or len(blob)<28 or blob[:4]!=b"GSW2": continue
                await broadcast(blob)
            elif message.get("text"):
                try:
                    data=json.loads(message["text"])
                    if data.get("type")=="asset" and re.fullmatch(r"[a-zA-Z0-9_-]{1,80}",str(data.get("id",""))) and (AVATARS/data["id"]/"avatar.json").is_file():
                        room["asset"]=data["id"]; await broadcast({"type":"asset","id":data["id"]})
                except (ValueError,TypeError): pass
    except WebSocketDisconnect:
        pass
    finally:
        room["clients"].discard(ws)
        if owner: room["owner"]=None; await broadcast({"type":"sender-left"})
        if not room["clients"]: rooms.pop(room_id,None)


if (WEB/"dist").is_dir(): app.mount("/",StaticFiles(directory=WEB/"dist",html=True),name="frontend")
