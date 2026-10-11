"""CDC stream: Debezium change events (Kafka) -> micro-batches -> Iceberg silver.

  app (Postgres) --WAL--> Debezium --> Kafka swingcoach.public.<table> --> this consumer
     per micro-batch: keep the last event per key -> parquet in s3://raw/cdc/<table>/
     -> MERGE upserts / DELETE deletes into lakehouse.silver -> refresh gold
     -> commit Kafka offsets only after the lake commit succeeded.

Delivery: at-least-once from Kafka + idempotent MERGE keyed by primary key with
"last state wins" => replaying a batch after a crash converges to the same tables.
Rows are shaped by extract.shape() with the same type contract as the batch path, so
the batch reconcile (pipeline/run.py) finds 0 differences when the stream is healthy.

Usage: uv run --no-sync python pipeline/stream.py [--once] [--wait 5]
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from confluent_kafka import Consumer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform  # noqa: E402
from extract import APP_TABLES, shape, to_arrow  # noqa: E402
from lake import RAW_BUCKET, Trino, clear_prefix, put_parquet, s3, trino_columns  # noqa: E402

TOPIC_PREFIX = "swingcoach.public."
SOURCE_KEYS = {"videos": ["id"], "swings": ["video_id"], "cards": ["video_id"], "reference": ["club_group"]}
SILVER_OF = {tbl: (target, keys) for target, (tbl, keys, _) in transform.SILVER.items() if tbl in APP_TABLES}


def consumer() -> Consumer:
    c = Consumer({
        "bootstrap.servers": "localhost:9094",
        "group.id": "swingcoach-lake",
        "enable.auto.commit": False,          # commit only after the lake write
        "auto.offset.reset": "earliest",
        # A crashed consumer holds its partitions until the session times out; keep that short.
        "session.timeout.ms": 10000,
    })
    c.subscribe([TOPIC_PREFIX + t for t in APP_TABLES])
    return c


def collapse(messages) -> tuple[dict[str, dict], int, int]:
    """Last event per (table, key), in log order.

    Returns ({table: {key: (op, row)}}, n_events, max capture lag in ms). Capture lag =
    Debezium processing time - Postgres commit time; both clocks are inside the WSL VM,
    which runs ~10 s apart from Windows, so mixing in this process's clock would be wrong.
    """
    latest: dict[str, dict] = {}
    n, max_lag = 0, 0
    for m in messages:
        if m.error() or m.value() is None:
            continue
        ev = json.loads(m.value())
        table = m.topic().removeprefix(TOPIC_PREFIX)
        op = ev["op"]                                   # r = snapshot, c / u / d
        row = ev["before"] if op == "d" else ev["after"]
        key = tuple(row[k] for k in SOURCE_KEYS[table])
        latest.setdefault(table, {})[key] = (op, row)
        n += 1
        if op != "r" and ev.get("ts_ms") and ev.get("source", {}).get("ts_ms"):
            max_lag = max(max_lag, ev["ts_ms"] - ev["source"]["ts_ms"])
    return latest, n, max_lag


def apply(t: Trino, client, table: str, events: dict, batch_id: str) -> tuple[int, int]:
    target, keys = SILVER_OF[table]
    ups = [row for op, row in events.values() if op != "d"]
    dels = [row for op, row in events.values() if op == "d"]
    frames = []
    if ups:
        frames.append(shape(table, pd.DataFrame(ups)).assign(_op="u"))
    if dels:
        frames.append(shape(table, pd.DataFrame(dels))[keys].assign(_op="d"))
    df = pd.concat(frames, ignore_index=True)
    arrow = to_arrow(df, table)
    clear_prefix(client, f"cdc/{table}/current/")
    put_parquet(client, arrow, f"cdc/{table}/current/part-0.parquet")
    put_parquet(client, arrow, f"cdc/{table}/batch={batch_id}/part-0.parquet")  # replay / audit

    cols_types = trino_columns(arrow.schema)
    t.run(f"DROP TABLE IF EXISTS raw.cdc.{table}")
    t.run(f"CREATE TABLE raw.cdc.{table} ({', '.join(f'{chr(34)}{c}{chr(34)} {ty}' for c, ty in cols_types)}) "
          f"WITH (external_location = 's3://{RAW_BUCKET}/cdc/{table}/current/', format = 'PARQUET')")
    fq = f"lakehouse.silver.{target}"
    upserted = deleted = 0
    if ups:
        cols = [c for c, _ in cols_types if c != "_op"]
        upserted = transform.upsert(t, fq, f"raw.cdc.{table}", cols, keys, where="_op = 'u'")
    if dels:
        key_tuple = ", ".join(f'"{k}"' for k in keys)
        cond = f"({key_tuple}) IN (SELECT {key_tuple} FROM raw.cdc.{table} WHERE _op = 'd')"
        deleted = t.scalar(f"SELECT count(*) FROM {fq} WHERE {cond}")
        if deleted:
            t.run(f"DELETE FROM {fq} WHERE {cond}")
    return upserted, deleted


def log_batch(t: Trino, row: dict) -> None:
    t.run("""CREATE TABLE IF NOT EXISTS lakehouse.ops.stream_batches (
               batch_id varchar, events integer, tables varchar, upserted integer, deleted integer,
               capture_lag_ms bigint, seconds double, committed_at timestamp(6) with time zone)""")
    t.run(f"""INSERT INTO lakehouse.ops.stream_batches VALUES ('{row["batch_id"]}', {row["events"]},
              '{json.dumps(row["tables"])}', {row["upserted"]}, {row["deleted"]}, {row["capture_lag_ms"]},
              {row["seconds"]:.2f}, current_timestamp)""")


def run(once: bool, wait: float) -> None:
    t, client, c = Trino(), s3(), consumer()
    t.run("CREATE SCHEMA IF NOT EXISTS raw.cdc WITH (location = 's3://raw/cdc/')")
    idle = 0
    try:
        while True:
            messages = c.consume(num_messages=5000, timeout=wait)
            latest, n, lag = collapse(messages)
            if not latest:
                # Only count idle polls once partitions are assigned (after a crash the
                # group needs a rebalance before anything can be read).
                if c.assignment():
                    idle += 1
                if once and idle >= 2:
                    break
                continue
            idle = 0
            t0 = time.perf_counter()
            batch_id = datetime.now().strftime("%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:6]
            stats = {table: apply(t, client, table, events, batch_id) for table, events in latest.items()}
            ups, dels = sum(s[0] for s in stats.values()), sum(s[1] for s in stats.values())
            if ups or dels:
                transform.build_gold(t)
            if os.environ.get("STREAM_CRASH_BEFORE_COMMIT"):  # fault injection for the recovery test
                print(f"batch {batch_id}: applied {stats}, crashing before offset commit", flush=True)
                os._exit(1)
            c.commit(asynchronous=False)  # only now: the lake already holds this batch
            log_batch(t, {"batch_id": batch_id, "events": n, "tables": {k: list(v) for k, v in stats.items()},
                          "upserted": ups, "deleted": dels, "capture_lag_ms": lag,
                          "seconds": time.perf_counter() - t0})
            print(f"batch {batch_id}: events={n} upserted={ups} deleted={dels} "
                  f"capture_lag={lag / 1000:.1f}s took={time.perf_counter() - t0:.1f}s {stats}", flush=True)
    finally:
        c.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="drain what is in Kafka, then exit")
    ap.add_argument("--wait", type=float, default=5.0, help="micro-batch window in seconds")
    args = ap.parse_args()
    run(args.once, args.wait)
