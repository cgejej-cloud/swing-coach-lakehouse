"""FastAPI app: upload videos in order, a background worker analyses them, the feed shows cards.

Run: uv run --no-sync python scripts/serve.py   (http://localhost:8000)
"""

import hashlib
import json
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from swing_coach import engine, insights
from swing_coach.coach import BY_KEY
from swing_coach.store import Store, default_store

ROOT = engine.ROOT
UPLOADS = ROOT / "data" / "uploads"
FLIGHT = {0: "알 수 없음", 1: "스트레이트", 2: "슬라이스", 3: "훅"}
CLUB_KO = {"DRIVER": "드라이버", "IRON": "아이언", "WEDGE": "웨지", "WOOD": "우드", "PUTTER": "퍼터",
           "driver": "드라이버", "iron": "아이언"}
MAX_ATTEMPTS = 2

store = default_store()
app = FastAPI(title="Swing Coach")
sync = insights.Sync()


# --- worker ---------------------------------------------------------------
def worker_loop() -> None:
    processed = 0
    while True:
        v = store.claim_next()
        if v is None:
            if processed:
                # Queue drained after new work: push the batch to the lakehouse once.
                sync.start()
                processed = 0
            time.sleep(1.0)
            continue
        try:
            engine.process(store, dict(v))
            store.finish(v["id"], "done")
            processed += 1
        except Exception as e:  # noqa: BLE001 - keep the worker alive, record the failure
            retry = v["attempts"] + 1 < MAX_ATTEMPTS
            store.finish(v["id"], "queued" if retry else "failed", f"{type(e).__name__}: {e}")
            traceback.print_exc()


@app.on_event("startup")
def start_worker() -> None:
    threading.Thread(target=worker_loop, daemon=True).start()


# --- API ------------------------------------------------------------------
@app.post("/api/uploads")
async def upload(files: list[UploadFile] = File(...)):
    """Files are queued in the order given; one upload batch = one session."""
    UPLOADS.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    session_key = f"upload-{now:%Y%m%d}"
    results = []
    for i, f in enumerate(files):
        data = await f.read()
        digest = hashlib.sha256(data).hexdigest()
        path = UPLOADS / f"{digest[:16]}.mp4"
        if not path.exists():
            path.write_bytes(data)
        vid, created = store.add_video(
            content_hash=digest, path=str(path), source="upload", session_key=session_key,
            session_label=f"{now:%m.%d} 업로드", shot_at=f"{now:%Y-%m-%d %H:%M:%S}.{i:03d}")
        results.append({"video_id": vid, "file": f.filename, "duplicate": not created})
    return {"session": session_key, "videos": results}


# --- resumable upload (tus-like: create, then PATCH chunks at an offset) ---------------
PARTIAL = UPLOADS / "partial"
MAX_UPLOAD = 500 * 1024 * 1024


class UploadInit(BaseModel):
    filename: str
    size: int
    batch: str          # client batch id: files of one selection share it
    index: int          # position inside the batch -> shot order


def _meta(upload_id: str) -> dict:
    # Validate before touching the filesystem: ids are uuid4 hex, nothing else.
    if not (len(upload_id) == 32 and all(c in "0123456789abcdef" for c in upload_id)):
        raise HTTPException(404, "unknown upload")
    path = PARTIAL / f"{upload_id}.json"
    if not path.exists():
        raise HTTPException(404, "unknown upload")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/resumable")
def resumable_create(body: UploadInit):
    if not 0 < body.size <= MAX_UPLOAD:
        raise HTTPException(413, "file too large")
    PARTIAL.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    (PARTIAL / f"{upload_id}.part").write_bytes(b"")
    (PARTIAL / f"{upload_id}.json").write_text(json.dumps(body.model_dump()), encoding="utf-8")
    return {"upload_id": upload_id, "offset": 0}


@app.get("/api/resumable/{upload_id}")
def resumable_offset(upload_id: str):
    meta = _meta(upload_id)
    return {"offset": (PARTIAL / f"{upload_id}.part").stat().st_size, "size": meta["size"]}


@app.patch("/api/resumable/{upload_id}")
async def resumable_patch(upload_id: str, offset: int, request: Request):
    meta = _meta(upload_id)
    part = PARTIAL / f"{upload_id}.part"
    current = part.stat().st_size
    # Always consume the body: answering before reading it makes the server reset the connection.
    chunk = await request.body()
    if offset != current:  # client resumes from the server's offset
        return JSONResponse({"offset": current}, status_code=409)
    if current + len(chunk) > meta["size"]:
        raise HTTPException(400, "chunk beyond declared size")
    with part.open("ab") as f:
        f.write(chunk)
    current += len(chunk)
    if current < meta["size"]:
        return {"offset": current}

    data = part.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    final = UPLOADS / f"{digest[:16]}.mp4"
    if not final.exists():
        part.replace(final)
    else:
        part.unlink()
    (PARTIAL / f"{upload_id}.json").unlink()
    now = datetime.now()
    vid, created = store.add_video(
        content_hash=digest, path=str(final), source="upload", session_key=f"upload-{meta['batch']}",
        session_label=f"{now:%m.%d} 업로드", shot_at=f"{now:%Y-%m-%d %H:%M:%S}.{meta['index']:03d}")
    return {"offset": current, "done": True, "video_id": vid, "duplicate": not created}


@app.get("/api/insights")
def get_insights():
    try:
        lake = insights.lakehouse()
        available = True
    except Exception as e:  # noqa: BLE001 - lakehouse is optional for the app
        lake, available = {"error": f"{type(e).__name__}"}, False
    return {"lakehouse_available": available, **lake, "findings": insights.findings(store),
            "sync": {"running": sync.running, "last": sync.last}}


@app.post("/api/sync")
def start_sync():
    return {"started": sync.start(), "running": sync.running}


@app.get("/api/videos/{video_id}/status")
def status(video_id: int):
    with store.conn() as c:
        row = c.execute("SELECT id, status, error, attempts FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    return dict(row)


def _card_json(r) -> dict:
    title = [f"{r['hole']}H" if r["hole"] else None,
             CLUB_KO.get(r["club"] or r["club_group"] or "", None),
             f"{r['carry_m']:.0f}m" if r["carry_m"] else None,
             FLIGHT.get(r["ball_flight_code"]) if r["ball_flight_code"] else None]
    return {
        "video_id": r["id"], "seq_no": r["seq_no"], "status": r["status"], "error": r["error"],
        "title": " · ".join(t for t in title if t),
        "club_predicted": r["club_source"] == "predicted",
        "club_uncertain": r["club_source"] == "uncertain",
        "good": r["good"], "fix": r["fix"], "cue": r["cue"],
        "extra": json.loads(r["extra_json"] or "[]"), "confidence": r["confidence"],
        "image": f"/media/{Path(r['image_path']).relative_to('data').as_posix()}" if r["image_path"] else None,
        "is_reference": bool(r["is_ref"]),
    }


@app.get("/api/feed")
def feed(limit_sessions: int = 10, before: str | None = None):
    with store.conn() as c:
        sessions = c.execute("""
            SELECT session_key, MAX(session_label) AS label, MAX(shot_at) AS last_at, COUNT(*) AS n
            FROM videos GROUP BY session_key
            HAVING (CAST(? AS TEXT) IS NULL OR MAX(shot_at) < CAST(? AS TEXT))
            ORDER BY last_at DESC LIMIT ?""", (before, before, limit_sessions)).fetchall()
        out = []
        for s in sessions:
            rows = c.execute("""
                SELECT v.*, c.good, c.fix, c.cue, c.extra_json, c.confidence, c.image_path,
                       EXISTS(SELECT 1 FROM reference r WHERE r.video_id = v.id) AS is_ref
                FROM videos v LEFT JOIN cards c ON c.video_id = v.id
                WHERE v.session_key = ? ORDER BY v.seq_no""", (s["session_key"],)).fetchall()
            out.append({"session": s["session_key"], "label": s["label"], "last_at": s["last_at"],
                        "cards": [_card_json(r) for r in rows]})
    return {"sessions": out, "next_before": out[-1]["last_at"] if out else None}


@app.get("/api/cards/{video_id}")
def card_detail(video_id: int):
    with store.conn() as c:
        r = c.execute("SELECT evidence_json, fix_key, good_key FROM cards WHERE video_id = ?", (video_id,)).fetchone()
    if not r:
        raise HTTPException(404)
    ev = json.loads(r["evidence_json"] or "{}")
    rows = [{"key": k, "name": BY_KEY[k].name, "phase": BY_KEY[k].phase, **v,
             "is_fix": k == r["fix_key"], "is_good": k == r["good_key"]} for k, v in ev.items()]
    return {"metrics": rows, "not_measurable": ["스윙 플레인", "클럽 궤도·페이스", "척추각 유지"]}


class RefIn(BaseModel):
    video_id: int


@app.post("/api/reference")
def set_reference(body: RefIn):
    with store.conn() as c:
        r = c.execute("SELECT club_group FROM videos WHERE id = ?", (body.video_id,)).fetchone()
    if not r or not r["club_group"]:
        raise HTTPException(400, "클럽군을 알 수 없는 영상은 기준으로 지정할 수 없어요")
    store.set_reference(r["club_group"], body.video_id)
    return {"club_group": r["club_group"], "video_id": body.video_id}


# Only card images are public; the database, poses and raw videos stay private.
(ROOT / "data" / "cards").mkdir(parents=True, exist_ok=True)
app.mount("/media/cards", StaticFiles(directory=ROOT / "data" / "cards"), name="cards")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")
