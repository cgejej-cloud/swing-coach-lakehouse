"""Batch run: extract -> register raw -> silver MERGE -> gold -> quality checks.

Full run (default) = reconciliation: re-lands every table and MERGEs it; with the CDC
stream healthy it should change nothing in the app tables (`upserted=0`).
`--only pose_frame` = the light run the app triggers after uploads (pose frames are
files, not DB rows, so they never travel through CDC).
Each run is logged to lakehouse.ops.pipeline_runs. Exit code 1 if any quality check fails.
Usage: uv run --no-sync python pipeline/run.py [--only pose_frame]
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402
import quality  # noqa: E402
import transform  # noqa: E402
from lake import Trino  # noqa: E402


def main(only: list[str] | None = None) -> int:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:6]
    t0 = time.perf_counter()
    t = Trino()
    landed = extract.main(only)
    transform.register_raw(t, only)
    silver = transform.merge_silver(t, only)
    gold = transform.build_gold(t)
    results = quality.checks(t)
    if only:
        # Landing snapshots of the other tables are stale in a partial run; check only what moved.
        targets = {tgt for tgt, (tbl, _, _) in transform.SILVER.items() if tbl in only}
        results = [r for r in results if r[0] in targets]
    quality.record(t, run_id, results)

    # Checks named "warn:..." are recorded but do not fail the run.
    failed = [r for r in results if not r[2] and not r[1].startswith("warn:")]
    for tb, name, ok, detail in results:
        label = "PASS" if ok else ("WARN" if name.startswith("warn:") else "FAIL")
        print(f"  [{label}] {tb}.{name}: {detail}")
    status = "failed" if failed else "succeeded"
    t.run("""CREATE TABLE IF NOT EXISTS lakehouse.ops.pipeline_runs (
               run_id varchar, status varchar, seconds double, landed varchar, silver varchar, gold varchar,
               dq_failed integer, finished_at timestamp(6) with time zone)""")
    t.run(f"""INSERT INTO lakehouse.ops.pipeline_runs VALUES ('{run_id}', '{status}',
              {time.perf_counter() - t0:.1f}, '{json.dumps(landed)}', '{json.dumps(silver)}',
              '{json.dumps(gold)}', {len(failed)}, current_timestamp)""")
    warned = sum(1 for r in results if not r[2] and r[1].startswith("warn:"))
    passed = sum(1 for r in results if r[2])
    print(f"run {run_id}: {status} in {time.perf_counter() - t0:.1f}s, "
          f"dq passed={passed} warnings={warned} failed={len(failed)} of {len(results)}")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="landing tables to run (default: all)")
    sys.exit(main(ap.parse_args().only))
