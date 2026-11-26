"""Alama dashboard - FastAPI backend.

Run with:  python run.py   then open http://localhost:8000
"""
import base64
import csv
import io
import json
import os
import re
import socket
import threading
import time
import uuid
import zipfile
from collections import Counter, deque
from html import escape
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import DB
from .detector import DEFAULT_NAMES, MockDetector, YoloDetector, draw_detections

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
DATA = BASE / "data"
FRAMES = DATA / "frames"
UPLOADS = DATA / "uploads"
for p in (MODELS, FRAMES, UPLOADS, MODELS / "assets"):
    p.mkdir(parents=True, exist_ok=True)

db = DB(DATA / "history.db")
det_lock = threading.Lock()          # one prediction at a time on the GPU
settings = {"conf": 0.40, "cooldown": 3.0, "save": True}


# ------------------------------------------------------------------ model handling
class State:
    detector = None
    model_name = ""
    demo = False
    load_error = ""
    esp32 = {"n": 0, "ts": 0.0, "jpeg": b"", "result": None}


state = State()


def metrics_path(model_name):
    return MODELS / (Path(model_name).stem + ".metrics.json")


def known_class_names():
    for f in sorted(MODELS.glob("*.metrics.json")):
        try:
            names = json.loads(f.read_text(encoding="utf-8")).get("class_names")
            if names:
                return names
        except Exception:
            pass
    return DEFAULT_NAMES


def load_model(name):
    """Load models/<name> (or switch to demo mode when name == '__demo__')."""
    with det_lock:
        if name == "__demo__":
            state.detector, state.model_name, state.demo = MockDetector(known_class_names()), "DEMO (mock)", True
            return
        det = YoloDetector(MODELS / name)
        state.detector, state.model_name, state.demo, state.load_error = det, name, False, ""


def init_model():
    pts = sorted(p.name for p in MODELS.glob("*.pt"))
    want = os.environ.get("MODEL")
    if os.environ.get("MOCK") == "1":
        load_model("__demo__"); return
    pick = want if want in pts else next((n for n in ("best_v1.pt", "best.pt") if n in pts), pts[0] if pts else None)
    if not pick:
        state.load_error = "No .pt file found in the models/ folder"
        load_model("__demo__"); return
    try:
        load_model(pick)
    except Exception as e:  # keep the dashboard usable and show the reason
        state.load_error = f"Could not load {pick}: {e}"
        print("MODEL LOAD ERROR:", e)
        load_model("__demo__")


# ------------------------------------------------------------------ tracking / stats
class Tracker:
    """Per-source memory: save cooldown + 'stable' speed limit (seen in >=3 of the last 5 frames)."""

    def __init__(self):
        self.last_saved = {}
        self.hist = deque(maxlen=5)

    def should_save(self, labels, t, cooldown):
        fire = [l for l in labels if t - self.last_saved.get(l, -1e9) >= cooldown]
        if fire:
            for l in labels:
                self.last_saved[l] = t
        return bool(fire)

    def stable_speed(self, dets):
        sp = next((d.label for d in dets if "speed" in d.label.lower()), None)
        self.hist.append(sp)
        c = Counter(x for x in self.hist if x)
        if c:
            lab, n = c.most_common(1)[0]
            if n >= 3:
                return lab
        return None


trackers = {}


def get_tracker(source):
    return trackers.setdefault(source, Tracker())


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.started = time.time()
        self.sources = {}
        self.classes = Counter()
        self.conf_sum = Counter()
        self.lat = deque(maxlen=300)
        self.stamps = deque(maxlen=60)

    def add(self, source, dets, ms, saved):
        with self.lock:
            s = self.sources.setdefault(source, {"total": 0, "with_det": 0, "saved": 0})
            s["total"] += 1
            s["with_det"] += 1 if dets else 0
            s["saved"] += 1 if saved else 0
            for d in dets:
                self.classes[d.label] += 1
                self.conf_sum[d.label] += d.conf
            self.lat.append(ms)
            self.stamps.append(time.time())

    def snapshot(self):
        with self.lock:
            tot = sum(s["total"] for s in self.sources.values())
            withd = sum(s["with_det"] for s in self.sources.values())
            saved = sum(s["saved"] for s in self.sources.values())
            fps = 0.0
            if len(self.stamps) > 1 and time.time() - self.stamps[-1] < 3:
                span = self.stamps[-1] - self.stamps[0]
                fps = (len(self.stamps) - 1) / span if span > 0 else 0.0
            lat = sorted(self.lat)
            return {
                "uptime_s": int(time.time() - self.started),
                "frames_total": tot, "frames_with_signs": withd, "frames_ignored": tot - withd, "frames_saved": saved,
                "fps": round(fps, 1),
                "latency_avg_ms": round(sum(lat) / len(lat), 1) if lat else 0,
                "latency_p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1) if len(lat) > 1 else (round(lat[0], 1) if lat else 0),
                "by_source": self.sources,
                "classes": [{"label": k, "n": v, "avg_conf": round(self.conf_sum[k] / v, 3)} for k, v in self.classes.most_common()],
            }


stats = Stats()


# ------------------------------------------------------------------ core processing
def decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the image (send a JPEG or PNG)")
    return img


def det_dict(d):
    return {"label": d.label, "cls_id": d.cls_id, "conf": round(d.conf, 4),
            "x1": round(d.x1, 4), "y1": round(d.y1, 4), "x2": round(d.x2, 4), "y2": round(d.y2, 4)}


def run_detect(img, source, mode="auto", tracker=None, now=None, video_t=None):
    """mode: auto = save with cooldown | dets = save if any sign | always | none."""
    tracker = tracker or get_tracker(source)
    now = time.time() if now is None else now
    t0 = time.perf_counter()
    with det_lock:
        dets = state.detector.predict(img, settings["conf"])
    ms = (time.perf_counter() - t0) * 1000
    save = False
    if mode == "always":
        save = True
    elif mode == "dets":
        save = bool(dets)
    elif mode == "auto":
        save = bool(dets) and settings["save"] and tracker.should_save([d.label for d in dets], now, settings["cooldown"])
    fid = None
    if save:
        fid = store_frame(img, dets, source, video_t)
    stable = tracker.stable_speed(dets)
    stats.add(source, dets, ms, fid is not None)
    h, w = img.shape[:2]
    return {"width": w, "height": h, "detections": [det_dict(d) for d in dets], "ms": round(ms, 1),
            "saved_frame_id": fid, "stable_speed": stable, "model": state.model_name}


def store_frame(img, dets, source, video_t=None):
    ts = time.time()
    name = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(ts))}_{uuid.uuid4().hex[:6]}.jpg"
    path = FRAMES / name
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    h, w = img.shape[:2]
    return db.add_frame(ts, source, state.model_name, w, h, dets, name, video_t)


def process_bytes(data, source, mode):
    return run_detect(decode(data), source, mode)


# ------------------------------------------------------------------ app
app = FastAPI(title="Alama")


@app.on_event("startup")
def _startup():
    init_model()
    print(f"\n  Model: {state.model_name}  |  device: {state.detector.device}")
    if state.demo:
        print("  *** DEMO MODE: results are FAKE. " + (state.load_error or "MOCK=1 set") + " ***")
    print("  Dashboard: http://localhost:8000\n")


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


def local_ips():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.append(s.getsockname()[0]); s.close()
    except Exception:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    return ips


@app.get("/api/info")
def info():
    pts = sorted(p.name for p in MODELS.glob("*.pt"))
    return {"model": state.model_name, "demo": state.demo, "load_error": state.load_error,
            "device": state.detector.device, "classes": state.detector.names, "models": pts,
            "ips": local_ips(), "settings": settings}


# ---- settings / stats
class SettingsIn(BaseModel):
    conf: Optional[float] = None
    cooldown: Optional[float] = None
    save: Optional[bool] = None


@app.post("/api/settings")
def set_settings(s: SettingsIn):
    if s.conf is not None:
        settings["conf"] = min(max(s.conf, 0.05), 0.95)
    if s.cooldown is not None:
        settings["cooldown"] = min(max(s.cooldown, 0.0), 60.0)
    if s.save is not None:
        settings["save"] = s.save
    return settings


@app.get("/api/stats")
def get_stats():
    return {"session": stats.snapshot(), "db": db.totals(), "settings": settings,
            "model": state.model_name, "demo": state.demo}


@app.post("/api/stats/reset")
def reset_stats():
    stats.reset(); trackers.clear()
    return {"ok": True}


# ---- models
class ModelIn(BaseModel):
    name: str


@app.post("/api/models/select")
def select_model(m: ModelIn):
    if m.name != "__demo__" and not (MODELS / m.name).exists():
        raise HTTPException(404, "Model file not found")
    try:
        load_model(m.name)
    except Exception as e:
        raise HTTPException(500, f"Could not load model: {e}")
    trackers.clear()
    return {"model": state.model_name, "demo": state.demo}


@app.get("/api/model-metrics")
def model_metrics():
    f = metrics_path(state.model_name) if not state.demo else None
    data = None
    if f and f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
    elif state.demo:  # show the real v1 numbers next to the demo banner so the tab can be previewed
        for g in sorted(MODELS.glob("*.metrics.json")):
            data = json.loads(g.read_text(encoding="utf-8")); break
    assets = {n: f"/api/model-asset/{n}" for n in ("confusion_matrix.png", "results.png", "PR_curve.png")
              if (MODELS / "assets" / n).exists()}
    return {"metrics": data, "assets": assets, "model": state.model_name}


@app.get("/api/model-asset/{name}")
def model_asset(name: str):
    p = (MODELS / "assets" / Path(name).name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# ---- detection endpoints
@app.post("/api/frame")
async def api_frame(request: Request, source: str = "upload", mode: str = "auto"):
    """Body = raw JPEG/PNG bytes. Used by the webcam and the image upload."""
    if source not in ("webcam", "upload", "compare", "esp32", "video"):
        raise HTTPException(400, "bad source")
    if mode not in ("auto", "dets", "always", "none"):
        raise HTTPException(400, "bad mode")
    data = await request.body()
    return await run_in_threadpool(process_bytes, data, source, mode)


class B64In(BaseModel):
    image: str  # base64 image, as in the lab handout


@app.post("/detect")
async def detect_lab(req: B64In):
    """Lab-compatible endpoint: base64 in -> objects + processing_time out (nothing is saved)."""
    raw = req.image.split(",")[-1]
    try:
        data = base64.b64decode(raw)
    except Exception:
        raise HTTPException(400, "image is not valid base64")
    t0 = time.perf_counter()
    res = await run_in_threadpool(process_bytes, data, "upload", "none")
    W, H = res["width"], res["height"]
    objs = [{"label": d["label"], "confidence": d["conf"],
             "bbox": [round(d["x1"] * W), round(d["y1"] * H), round(d["x2"] * W), round(d["y2"] * H)]} for d in res["detections"]]
    return {"objects": objs, "processing_time": round(time.perf_counter() - t0, 4)}


@app.post("/ingest")
async def ingest(request: Request):
    """ESP32-CAM endpoint: POST a raw JPEG (Content-Type: image/jpeg). Returns a SMALL json reply."""
    data = await request.body()
    res = await run_in_threadpool(process_bytes, data, "esp32", "auto")
    state.esp32 = {"n": state.esp32["n"] + 1, "ts": time.time(), "jpeg": data, "result": res}
    sp = res["stable_speed"]
    m = re.search(r"\d+", sp) if sp else None
    return {"ok": True, "count": len(res["detections"]), "speed": int(m.group()) if m else None,
            "alert": sp is not None, "ms": res["ms"],
            "objects": [{"label": d["label"], "conf": round(d["conf"], 2)} for d in res["detections"][:3]]}


@app.get("/api/esp32/latest.json")
def esp32_latest():
    e = state.esp32
    return {"n": e["n"], "age_s": round(time.time() - e["ts"], 1) if e["ts"] else None, "result": e["result"]}


@app.get("/api/esp32/latest.jpg")
def esp32_latest_jpg():
    if not state.esp32["jpeg"]:
        raise HTTPException(404, "no frame received yet")
    return Response(state.esp32["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---- video upload (processed in a background thread)
jobs = {}


def run_video(job_id, path, sample_fps):
    job = jobs[job_id]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        job.update(status="error", error="Could not open this video file"); return
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, round(vfps / max(sample_fps, 0.5)))
    tracker, idx = Tracker(), 0
    job.update(status="running", duration=round(total / vfps, 1) if total else None)
    try:
        while not job.get("cancel"):
            if not cap.grab():
                break
            if idx % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    t = idx / vfps
                    res = run_detect(img, "video", "auto", tracker, now=t, video_t=t)
                    job["processed"] += 1
                    if res["detections"]:
                        job["with_signs"] += 1
                    if res["saved_frame_id"]:
                        top = res["detections"][0]
                        job["events"].append({"t": round(t, 1), "label": top["label"], "conf": top["conf"],
                                              "n": len(res["detections"]), "frame_id": res["saved_frame_id"]})
                    job["last_frame_id"] = res["saved_frame_id"] or job["last_frame_id"]
            idx += 1
            job["progress"] = round(idx / total, 3) if total else 0
        job["status"] = "cancelled" if job.get("cancel") else "done"
        job["progress"] = 1.0 if job["status"] == "done" else job["progress"]
    except Exception as e:
        job.update(status="error", error=str(e))
    finally:
        cap.release()


@app.post("/api/video")
def upload_video(file: UploadFile = File(...), sample_fps: float = Form(5.0)):
    ext = Path(file.filename or "v.mp4").suffix.lower() or ".mp4"
    job_id = uuid.uuid4().hex[:8]
    path = UPLOADS / f"{job_id}{ext}"
    with open(path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            f.write(chunk)
    jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0, "processed": 0, "with_signs": 0,
                    "events": [], "last_frame_id": None, "filename": file.filename, "sample_fps": sample_fps}
    threading.Thread(target=run_video, args=(job_id, path, sample_fps), daemon=True).start()
    return {"job": job_id}


@app.get("/api/video/{job_id}")
def video_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]


@app.post("/api/video/{job_id}/cancel")
def video_cancel(job_id: str):
    if job_id in jobs:
        jobs[job_id]["cancel"] = True
    return {"ok": True}


# ---- history
def filters(label: Optional[str] = None, source: Optional[str] = None, model: Optional[str] = None,
            flag: Optional[str] = None, min_conf: Optional[float] = None,
            date_from: Optional[str] = None, date_to: Optional[str] = None):
    return {k: v for k, v in dict(label=label, source=source, model=model, flag=flag, min_conf=min_conf,
                                  date_from=date_from, date_to=date_to).items() if v not in (None, "", 0)}


@app.get("/api/history")
def history(label: Optional[str] = None, source: Optional[str] = None, model: Optional[str] = None,
            flag: Optional[str] = None, min_conf: Optional[float] = None, date_from: Optional[str] = None,
            date_to: Optional[str] = None, limit: int = Query(40, le=200), offset: int = 0):
    f = filters(label, source, model, flag, min_conf, date_from, date_to)
    total, items = db.history(f, limit, offset)
    return {"total": total, "items": items}


@app.get("/api/frame/{fid}")
def frame_info(fid: int):
    fr = db.get_frame(fid)
    if not fr:
        raise HTTPException(404)
    return fr


@app.get("/api/frame/{fid}/image")
def frame_image(fid: int):
    fr = db.get_frame(fid)
    if not fr or not (FRAMES / fr["path"]).exists():
        raise HTTPException(404)
    return FileResponse(FRAMES / fr["path"])


@app.get("/api/frame/{fid}/annotated")
def frame_annotated(fid: int, w: int = 640):
    fr = db.get_frame(fid)
    if not fr or not (FRAMES / fr["path"]).exists():
        raise HTTPException(404)
    img = cv2.imread(str(FRAMES / fr["path"]))
    if w and img.shape[1] > w:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
    wrong = {d["id"] for d in fr["detections"] if d["wrong"]}
    img = draw_detections(img, fr["detections"], wrong)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg")


class FlagIn(BaseModel):
    flag: Optional[str] = None
    note: Optional[str] = None


@app.post("/api/frame/{fid}/flag")
def flag_frame(fid: int, f: FlagIn):
    if f.flag not in (None, "ok", "missed"):
        raise HTTPException(400, "flag must be ok, missed or null")
    if not db.get_frame(fid):
        raise HTTPException(404)
    db.set_frame_flag(fid, f.flag, f.note)
    return {"ok": True}


class WrongIn(BaseModel):
    wrong: bool


@app.post("/api/detection/{did}/wrong")
def flag_detection(did: int, w: WrongIn):
    db.set_det_wrong(did, w.wrong)
    return {"ok": True}


@app.delete("/api/frame/{fid}")
def delete_frame(fid: int):
    name = db.delete_frame(fid)
    if name:
        (FRAMES / name).unlink(missing_ok=True)
    return {"ok": True}


# ---- compare two saved frames
def _best_by_label(fr):
    out = {}
    for d in fr["detections"]:
        if d["wrong"]:
            continue
        o = out.setdefault(d["label"], {"n": 0, "max": 0.0})
        o["n"] += 1
        o["max"] = max(o["max"], d["conf"])
    return out


@app.get("/api/compare")
def compare(a: int, b: int):
    fa, fb = db.get_frame(a), db.get_frame(b)
    if not fa or not fb:
        raise HTTPException(404, "frame not found")
    A, B = _best_by_label(fa), _best_by_label(fb)
    rows = []
    for lab in sorted(set(A) | set(B)):
        ca, cb = A.get(lab), B.get(lab)
        rows.append({"label": lab, "a_n": ca["n"] if ca else 0, "b_n": cb["n"] if cb else 0,
                     "a_conf": ca["max"] if ca else None, "b_conf": cb["max"] if cb else None,
                     "delta": round(cb["max"] - ca["max"], 4) if ca and cb else None,
                     "status": "both" if ca and cb else ("only_a" if ca else "only_b")})
    def summ(f):
        return {"id": f["id"], "ts_iso": f["ts_iso"], "source": f["source"], "model": f["model"],
                "n_det": len([d for d in f["detections"] if not d["wrong"]]), "max_conf": f["max_conf"],
                "size": f"{f['width']}x{f['height']}"}
    return {"a": summ(fa), "b": summ(fb), "rows": rows,
            "same_classes": {r["label"] for r in rows if r["status"] != "both"} == set()}


# ---- exports
@app.get("/api/history.csv")
def history_csv(label: Optional[str] = None, source: Optional[str] = None, model: Optional[str] = None,
                flag: Optional[str] = None, min_conf: Optional[float] = None,
                date_from: Optional[str] = None, date_to: Optional[str] = None):
    f = filters(label, source, model, flag, min_conf, date_from, date_to)
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["frame_id", "time", "source", "model", "video_time_s", "frame_flag", "label", "confidence",
                 "x1", "y1", "x2", "y2", "marked_wrong"])
    for fr in db.all_frames(f):
        for d in fr["detections"] or [None]:
            wr.writerow([fr["id"], fr["ts_iso"], fr["source"], fr["model"], fr["video_t"], fr["flag"] or ""] +
                        ([d["label"], d["conf"], d["x1"], d["y1"], d["x2"], d["y2"], d["wrong"]] if d else [""] * 7))
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=detections.csv"})


@app.get("/api/export/dataset.zip")
def export_dataset(label: Optional[str] = None, source: Optional[str] = None, model: Optional[str] = None,
                   flag: Optional[str] = None, min_conf: Optional[float] = None,
                   date_from: Optional[str] = None, date_to: Optional[str] = None):
    """Saved frames + YOLO label files (wrong detections removed) -> upload to Roboflow and correct."""
    f = filters(label, source, model, flag, min_conf, date_from, date_to)
    names = state.detector.names
    mem = io.BytesIO()
    review = []
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for fr in db.all_frames(f):
            p = FRAMES / fr["path"]
            if not p.exists():
                continue
            stem = f"frame_{fr['id']:06d}"
            z.write(p, f"images/{stem}.jpg")
            lines = []
            for d in fr["detections"]:
                if d["wrong"]:
                    continue
                cx, cy = (d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2
                lines.append(f"{d['cls_id']} {cx:.6f} {cy:.6f} {d['x2'] - d['x1']:.6f} {d['y2'] - d['y1']:.6f}")
            z.writestr(f"labels/{stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
            if fr["flag"] == "missed" or any(d["wrong"] for d in fr["detections"]):
                review.append(stem)
        z.writestr("data.yaml", "names: [" + ", ".join(json.dumps(n) for n in names) + f"]\nnc: {len(names)}\n")
        z.writestr("needs_review.txt", "These frames were flagged (missed sign / wrong detection). Fix their labels in Roboflow:\n" + "\n".join(review) + "\n")
        z.writestr("README.txt", "Pre-labelled frames from the dashboard (labels = model predictions, wrong ones removed).\n"
                   "Upload the images AND labels to Roboflow, then check/correct every box before training.\n")
    mem.seek(0)
    return StreamingResponse(mem, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=dashboard_dataset.zip"})


@app.get("/api/report", response_class=HTMLResponse)
def report():
    """Printable session report (open it, then Ctrl+P -> Save as PDF)."""
    s, t = stats.snapshot(), db.totals()
    mm = model_metrics()["metrics"]
    _, recent = db.history({}, 12, 0)
    rows = "".join(f"<tr><td>{escape(c['label'])}</td><td>{c['n']}</td><td>{c['avg_conf']}</td></tr>" for c in t["by_class"])
    mrows = ""
    if mm:
        o = mm["overall"]
        mrows = (f"<p>Measured on the <b>{escape(mm.get('split', 'test'))}</b> set: precision {o['precision']:.2f}, "
                 f"recall {o['recall']:.2f}, mAP50 {o['map50']:.2f}, mAP50-95 {o['map50_95']:.2f}.</p><table><tr><th>Class</th><th>Instances</th><th>P</th><th>R</th><th>mAP50</th></tr>" +
                 "".join(f"<tr><td>{escape(c['name'])}</td><td>{c['instances']}</td><td>{c['precision']:.2f}</td><td>{c['recall']:.2f}</td><td>{c['map50']:.2f}</td></tr>" for c in mm["classes"]) + "</table>")
    imgs = "".join(f"<figure><img src='/api/frame/{r['id']}/annotated?w=480'><figcaption>#{r['id']} {escape(r['ts_iso'])} - {escape(r['source'])}</figcaption></figure>" for r in recent)
    return f"""<!doctype html><meta charset=utf-8><title>Session report</title>
<style>body{{font:14px system-ui;max-width:900px;margin:30px auto;color:#111}}table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #bbb;padding:4px 10px;text-align:left}}
figure{{display:inline-block;margin:6px;width:280px}}img{{width:100%}}h1,h2{{margin:18px 0 6px}}@media print{{button{{display:none}}}}</style>
<button onclick="print()">Print / Save as PDF</button>
<h1>Speed Sign Detection - session report</h1><p>Generated {time.strftime('%Y-%m-%d %H:%M')} - model <b>{escape(state.model_name)}</b>{' (DEMO, fake results)' if state.demo else ''}</p>
<h2>This session</h2><p>Frames processed: <b>{s['frames_total']}</b> - with signs: <b>{s['frames_with_signs']}</b> - ignored (empty): <b>{s['frames_ignored']}</b> - saved: <b>{s['frames_saved']}</b><br>
Avg latency: <b>{s['latency_avg_ms']} ms</b> - p95: <b>{s['latency_p95_ms']} ms</b></p>
<h2>History totals</h2><p>Saved frames: {t['frames_saved']} - detections: {t['detections_saved']} - marked wrong: {t['wrong_marked']} - missed signs flagged: {t['missed_marked']}</p>
<table><tr><th>Class</th><th>Detections</th><th>Avg confidence</th></tr>{rows}</table>
<h2>Model quality</h2>{mrows or '<p>No metrics file for this model.</p>'}
<h2>Latest saved frames</h2>{imgs}"""
