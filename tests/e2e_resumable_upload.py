"""End-to-end check against a running app (scripts/serve.py): resumable upload with a
retried chunk and a resume, then the worker's analysis.

Usage: uv run --no-sync python tests/e2e_resumable_upload.py <video.mp4>
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
CHUNK = 200_000


def call(method: str, path: str, body=None, ctype: str = "application/json"):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main(path: str) -> None:
    data = open(path, "rb").read()
    st, init = call("POST", "/api/resumable",
                    {"filename": path, "size": len(data), "batch": time.strftime("%Y%m%d%H%M%S"), "index": 0})
    uid = init["upload_id"]
    st, r = call("PATCH", f"/api/resumable/{uid}?offset=0", data[:CHUNK], "application/octet-stream")
    assert st == 200 and r["offset"] == CHUNK, (st, r)
    # Response lost -> client re-sends the same chunk; server must refuse and report its offset.
    st, r = call("PATCH", f"/api/resumable/{uid}?offset=0", data[:CHUNK], "application/octet-stream")
    assert st == 409 and r["offset"] == CHUNK, (st, r)
    # Reconnect -> ask where to resume.
    st, r = call("GET", f"/api/resumable/{uid}")
    off = r["offset"]
    assert off == CHUNK
    while True:
        st, r = call("PATCH", f"/api/resumable/{uid}?offset={off}", data[off:off + CHUNK], "application/octet-stream")
        off = r["offset"]
        if r.get("done"):
            break
    print("upload ok:", r)
    t0 = time.time()
    while True:
        st, s = call("GET", f"/api/videos/{r['video_id']}/status")
        if s["status"] in ("done", "failed") or time.time() - t0 > 180:
            break
        time.sleep(1)
    print(f"processing: {s['status']} in {time.time() - t0:.0f}s", s.get("error") or "")
    st, feed = call("GET", "/api/feed?limit_sessions=1")
    card = feed["sessions"][0]["cards"][0]
    print({k: card[k] for k in ("title", "club_predicted", "club_uncertain", "good", "fix", "cue")})


if __name__ == "__main__":
    main(sys.argv[1])
