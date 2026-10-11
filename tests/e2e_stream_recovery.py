"""Crash between the lake write and the Kafka offset commit, then restart.

Expected: the restarted consumer re-reads the same events (at-least-once) and the
idempotent MERGE turns them into a no-op, leaving silver correct.
The stream consumer must NOT be running elsewhere in the same group.
Usage: uv run --no-sync python tests/e2e_stream_recovery.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from lake import ENV, Trino  # noqa: E402

STREAM = [sys.executable, str(ROOT / "pipeline" / "stream.py"), "--once", "--wait", "3"]


def run_stream(crash: bool) -> str:
    env = dict(os.environ, **({"STREAM_CRASH_BEFORE_COMMIT": "1"} if crash else {}))
    p = subprocess.run(STREAM, capture_output=True, text=True, encoding="utf-8", env=env)
    return "\n".join(l for l in p.stdout.splitlines() if l.startswith("batch"))


t = Trino()
pg = psycopg.connect(ENV["APP_DATABASE_URL"], autocommit=True)
run_stream(crash=False)  # start from a drained topic

carry = pg.execute("SELECT carry_m FROM videos WHERE id = 2").fetchone()[0]
pg.execute("UPDATE videos SET carry_m = carry_m + 7 WHERE id = 2")

first = run_stream(crash=True)
print("1st run :", first)
assert "crashing before offset commit" in first
lake = t.run("SELECT carry_m FROM lakehouse.silver.fact_video WHERE video_id = 2")[0][0]
assert lake == carry + 7, lake

second = run_stream(crash=False)
print("restart :", second)
events = int(re.search(r"events=(\d+)", second).group(1))
upserted = int(re.search(r"upserted=(\d+)", second).group(1))
assert events >= 1, "uncommitted events must be redelivered"
assert upserted == 0, "redelivery must be a no-op"
assert t.run("SELECT carry_m FROM lakehouse.silver.fact_video WHERE video_id = 2")[0][0] == carry + 7

pg.execute("UPDATE videos SET carry_m = %s WHERE id = 2", (carry,))
run_stream(crash=False)
assert t.run("SELECT carry_m FROM lakehouse.silver.fact_video WHERE video_id = 2")[0][0] == carry
print("recovery OK: redelivered", events, "event(s), applied 0 changes; value reverted")
