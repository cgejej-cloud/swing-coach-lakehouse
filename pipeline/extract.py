"""Batch extract: app DB tables and pose files -> S3 landing zone as parquet.

Every run writes a full snapshot to   landing/<table>/latest/   (what Trino reads)
and keeps a copy under                landing/<table>/dt=<run date>/   (replay / audit).

The app tables also stream through CDC (pipeline/stream.py); this batch path is the
reconciliation run and the only path for pose frames (files, not DB rows). Both
paths shape rows with the same `shape()` and type contract so their row hashes agree.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lake import LANDING, ROOT, clear_prefix, put_parquet, s3  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from swing_coach.store import default_store  # noqa: E402

POSE_DIR = ROOT / "data" / "pose"
APP_TABLES = ["videos", "swings", "cards", "reference"]
VIDEO_COLUMNS = ["video_id", "source", "nasmo_seq", "session_key", "session_label", "seq_no", "shot_at",
                 "uploaded_at", "status", "error", "club", "club_group", "club_source", "club_conf",
                 "ball_flight_code", "carry_m", "hole", "width", "fps", "content_hash"]

# Data contract: integer columns stay integers even when a batch has NULLs in them.
# Without it pandas infers float for a column that happens to contain a NULL (e.g. an
# uploaded video has no nasmo_seq), and the silver MERGE fails on bigint vs double.
INT_COLUMNS = {
    "videos": ["video_id", "nasmo_seq", "seq_no", "ball_flight_code", "hole", "width"],
    "swings": ["video_id", "p1", "p4", "p7", "p10", "m_scale_ok"],
    "cards": ["video_id"],
    "reference": ["video_id"],
}
FLOAT_COLUMNS = {
    "videos": ["club_conf", "carry_m", "fps"],
}


def shape(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Source rows (app DB columns) -> landing columns. Shared by batch and CDC paths."""
    if name == "videos":
        return df.rename(columns={"id": "video_id"}).reindex(columns=VIDEO_COLUMNS)
    if name == "swings":
        df = df.reset_index(drop=True)
        metrics = pd.DataFrame([json.loads(m) if isinstance(m, str) else {} for m in df.pop("metrics_json")])
        metrics = metrics.drop(columns=["feature_version"], errors="ignore").add_prefix("m_")
        return pd.concat([df, metrics], axis=1)
    if name == "cards":
        return df.drop(columns=["image_path"], errors="ignore")  # local file path, not data
    return df


def read_app_tables() -> dict[str, pd.DataFrame]:
    store = default_store()
    with store.conn() as c:
        raw = {t: pd.DataFrame([dict(r) for r in c.execute(f"SELECT * FROM {t} ORDER BY 1")]) for t in APP_TABLES}
    return {t: shape(t, df) for t, df in raw.items()}


def read_pose_frames(videos: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for v in videos.itertuples():
        name = f"{int(v.nasmo_seq)}.parquet" if pd.notna(v.nasmo_seq) else f"h{v.content_hash[:16]}.parquet"
        path = POSE_DIR / name
        if path.exists():
            p = pd.read_parquet(path, columns=["frame", "t_ms", "lm", "x", "y", "z", "vis"])
            parts.append(p.assign(video_id=v.video_id))
    df = pd.concat(parts, ignore_index=True)
    return df.astype({"frame": "int32", "t_ms": "int32", "lm": "int8", "x": "float32", "y": "float32",
                      "z": "float32", "vis": "float32", "video_id": "int64"})


def to_arrow(df: pd.DataFrame, name: str) -> pa.Table:
    df = df.copy()
    for c in INT_COLUMNS.get(name, []):
        if c in df:
            df[c] = pd.to_numeric(df[c]).astype("Int64")
    for c in FLOAT_COLUMNS.get(name, []):
        if c in df:
            df[c] = pd.to_numeric(df[c]).astype("float64")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    return pa.Table.from_pandas(df, preserve_index=False)


def main(only: list[str] | None = None) -> dict[str, int]:
    client = s3()
    run_dt = datetime.now().strftime("%Y-%m-%d")
    tables = read_app_tables()
    tables["pose_frame"] = read_pose_frames(tables["videos"])
    counts = {}
    for name, df in tables.items():
        if only and name not in only:
            continue
        t = to_arrow(df, name)
        clear_prefix(client, f"{LANDING}/{name}/latest/")
        put_parquet(client, t, f"{LANDING}/{name}/latest/part-0.parquet")
        put_parquet(client, t, f"{LANDING}/{name}/dt={run_dt}/part-0.parquet")
        counts[name] = t.num_rows
        print(f"extract {name}: {t.num_rows} rows")
    return counts


if __name__ == "__main__":
    main()
