"""End-to-end CDC check: app/Postgres change -> Debezium -> Kafka -> stream -> Iceberg silver.

Needs the platform up, the app (scripts/serve.py) and the stream (pipeline/stream.py) running.
Every change made here is reverted at the end.
Usage: uv run --no-sync python tests/e2e_cdc.py
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from lake import ENV, Trino  # noqa: E402

t = Trino()
pg = psycopg.connect(ENV["APP_DATABASE_URL"], autocommit=True)


def wait_for(sql: str, expected, timeout: float = 60.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        rows = t.run(sql)
        value = rows[0][0] if rows else None
        if value == expected:
            return time.time() - t0
        time.sleep(0.5)
    raise AssertionError(f"timeout: {sql} != {expected!r} (last {value!r})")


def api(method: str, path: str, body: dict) -> dict:
    req = urllib.request.Request("http://127.0.0.1:8000" + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


results = {}

# 1) App action: change the driver reference swing through the API.
old_ref = pg.execute("SELECT video_id FROM reference WHERE club_group = 'driver'").fetchone()[0]
new_ref = pg.execute("""SELECT id FROM videos WHERE club_group = 'driver' AND ball_flight_code = 1
                        AND id <> %s ORDER BY carry_m DESC LIMIT 1""", (old_ref,)).fetchone()[0]
api("POST", "/api/reference", {"video_id": new_ref})
results["app_reference_change"] = wait_for(
    "SELECT video_id FROM lakehouse.silver.dim_reference WHERE club_group = 'driver'", new_ref)
api("POST", "/api/reference", {"video_id": old_ref})
wait_for("SELECT video_id FROM lakehouse.silver.dim_reference WHERE club_group = 'driver'", old_ref)

# 2) In-place value change with no timestamp touched (the incremental-load blind spot).
carry = pg.execute("SELECT carry_m FROM videos WHERE id = 1").fetchone()[0]
pg.execute("UPDATE videos SET carry_m = carry_m + 1 WHERE id = 1")
results["silent_update"] = wait_for("SELECT carry_m FROM lakehouse.silver.fact_video WHERE video_id = 1", carry + 1)
pg.execute("UPDATE videos SET carry_m = %s WHERE id = 1", (carry,))
wait_for("SELECT carry_m FROM lakehouse.silver.fact_video WHERE video_id = 1", carry)

# 3) Insert then delete a row: the delete must reach the lake too.
vid = pg.execute("""INSERT INTO videos (content_hash, path, source, session_key, seq_no, status)
                    VALUES ('cdc-e2e-test', '-', 'upload', 'cdc-test', 1, 'failed') RETURNING id""").fetchone()[0]
results["insert"] = wait_for(f"SELECT count(*) FROM lakehouse.silver.fact_video WHERE video_id = {vid}", 1)
pg.execute("DELETE FROM videos WHERE id = %s", (vid,))
results["delete"] = wait_for(f"SELECT count(*) FROM lakehouse.silver.fact_video WHERE video_id = {vid}", 0)

for k, v in results.items():
    print(f"{k:22s} visible in Iceberg after {v:.1f}s")
print("all CDC scenarios passed; changes reverted")
