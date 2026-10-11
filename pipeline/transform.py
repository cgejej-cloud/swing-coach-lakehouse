"""Landing parquet -> Iceberg silver (MERGE by row hash) -> gold marts, all in Trino SQL.

Change detection does not rely on an "updated at" column: each silver row stores a
hash of its content, and MERGE only rewrites rows whose hash changed. Re-running on
unchanged input therefore writes nothing (idempotent), and a value changed in place
upstream is still caught - the failure mode of timestamp-based increments.
"""

import io
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lake import LANDING, RAW_BUCKET, Trino, s3, trino_columns  # noqa: E402

# silver table -> (landing table, key columns, partitioning)
SILVER = {
    "fact_video": ("videos", ["video_id"], []),
    "fact_swing": ("swings", ["video_id"], []),
    "fact_card": ("cards", ["video_id"], []),
    "dim_reference": ("reference", ["club_group"], []),
    "fact_pose_frame": ("pose_frame", ["video_id", "frame", "lm"], ["bucket(video_id, 8)"]),
}


def landing_schema(table: str):
    obj = s3().get_object(Bucket=RAW_BUCKET, Key=f"{LANDING}/{table}/latest/part-0.parquet")
    return pq.read_schema(io.BytesIO(obj["Body"].read()))


def register_raw(t: Trino, only: list[str] | None = None) -> None:
    """(Re)create external tables over landing/<table>/latest/. Dropping an external table keeps the files."""
    t.run("CREATE SCHEMA IF NOT EXISTS raw.landing WITH (location = 's3://raw/landing/')")
    for _, (table, _, _) in SILVER.items():
        if only and table not in only:
            continue
        cols = ", ".join(f'"{c}" {ty}' for c, ty in trino_columns(landing_schema(table)))
        t.run(f"DROP TABLE IF EXISTS raw.landing.{table}")
        t.run(f"CREATE TABLE raw.landing.{table} ({cols}) WITH ("
              f"external_location = 's3://raw/{LANDING}/{table}/latest/', format = 'PARQUET')")


def _hash_expr(cols: list[str]) -> str:
    parts = ", ".join(f"coalesce(cast(\"{c}\" AS varchar), '<null>')" for c in cols)
    return f"to_hex(xxhash64(to_utf8(concat_ws('|', {parts}))))"


NOW6 = "CAST(current_timestamp AS timestamp(6) with time zone)"


def upsert(t: Trino, fq: str, src_table: str, cols: list[str], keys: list[str], where: str = "") -> int:
    """MERGE rows of src_table (optionally filtered) into fq, rewriting only rows whose hash changed.
    Shared by the batch reconcile and the CDC stream so both compute identical row hashes."""
    filt = f" WHERE {where}" if where else ""
    select_cols = ", ".join(f'"{c}"' for c in cols)
    src = f"(SELECT {select_cols}, {_hash_expr(cols)} AS row_hash FROM {src_table}{filt})"
    on = " AND ".join(f't."{k}" = s."{k}"' for k in keys)
    changed = t.scalar(f"SELECT count(*) FROM {src} s LEFT JOIN {fq} t ON {on} "
                       f"WHERE t.row_hash IS NULL OR t.row_hash <> s.row_hash")
    if changed:
        all_cols = cols + ["row_hash"]
        t.run(f"""MERGE INTO {fq} t USING {src} s ON {on}
                  WHEN MATCHED AND t.row_hash <> s.row_hash THEN UPDATE SET
                    {", ".join(f'"{c}" = s."{c}"' for c in all_cols)}, _loaded_at = {NOW6}
                  WHEN NOT MATCHED THEN INSERT ({", ".join(f'"{c}"' for c in all_cols)}, _loaded_at)
                    VALUES ({", ".join(f's."{c}"' for c in all_cols)}, {NOW6})""")
    return changed


def merge_silver(t: Trino, only: list[str] | None = None) -> dict[str, dict]:
    t.run("CREATE SCHEMA IF NOT EXISTS lakehouse.silver")
    stats = {}
    for target, (table, keys, partitioning) in SILVER.items():
        if only and table not in only:
            continue
        cols = [c for c, _ in trino_columns(landing_schema(table))]
        src = f"(SELECT *, {_hash_expr(cols)} AS row_hash FROM raw.landing.{table})"
        fq = f"lakehouse.silver.{target}"
        part = f" WITH (partitioning = ARRAY[{', '.join(repr(p) for p in partitioning)}])" if partitioning else ""
        _rebuild_if_type_changed(t, fq, table)
        t.run(f"CREATE TABLE IF NOT EXISTS {fq}{part} AS "
              f"SELECT s.*, {NOW6} AS _loaded_at FROM {src} s WITH NO DATA")
        _add_new_columns(t, fq, table)
        changed = upsert(t, fq, f"raw.landing.{table}", cols, keys)
        # Full snapshots in landing: keys that disappeared upstream are deleted here.
        key_tuple = ", ".join(f'"{k}"' for k in keys)
        deleted = t.scalar(f"SELECT count(*) FROM {fq} WHERE ({key_tuple}) NOT IN "
                           f"(SELECT {key_tuple} FROM raw.landing.{table})")
        if deleted:
            t.run(f"DELETE FROM {fq} WHERE ({key_tuple}) NOT IN (SELECT {key_tuple} FROM raw.landing.{table})")
        stats[target] = {"upserted": changed, "deleted": deleted, "rows": t.scalar(f"SELECT count(*) FROM {fq}")}
        print(f"silver {target}: upserted={changed} deleted={deleted} rows={stats[target]['rows']}")
    return stats


def _add_new_columns(t: Trino, fq: str, table: str) -> None:
    """Schema evolution: a new metric in landing becomes a new nullable column in silver."""
    existing = {r[0] for r in t.run(f"DESCRIBE {fq}")}
    for c, ty in trino_columns(landing_schema(table)):
        if c not in existing:
            t.run(f'ALTER TABLE {fq} ADD COLUMN "{c}" {ty}')
            print(f"  schema evolution: {fq} + {c} {ty}")


def _rebuild_if_type_changed(t: Trino, fq: str, table: str) -> bool:
    """A column whose type differs from the landing contract (e.g. created as varchar while
    all values were NULL) cannot be altered in place (Iceberg only widens int->bigint,
    float->double). Silver is derived from a full landing snapshot, so drop and reload it."""
    try:
        existing = {r[0]: r[1] for r in t.run(f"DESCRIBE {fq}")}
    except Exception:  # table does not exist yet
        return False
    # Iceberg has no tinyint/smallint: Trino stores them as integer. Not a drift.
    stored = {"tinyint": "integer", "smallint": "integer"}
    drift = [(c, existing[c], ty) for c, ty in trino_columns(landing_schema(table))
             if c in existing and existing[c] != stored.get(ty, ty)]
    if drift:
        print(f"  type drift in {fq}: {drift} -> rebuilding from landing snapshot")
        t.run(f"DROP TABLE {fq}")
    return bool(drift)


DBT_DIR = Path(__file__).resolve().parents[1] / "dbt"
GOLD_MODELS = ["round_summary", "metric_by_flight", "fix_frequency", "monthly_trend"]


class GoldBuildError(RuntimeError):
    pass


def build_gold(t: Trino) -> dict[str, int]:
    """Rebuild the gold marts with dbt (dbt/models/gold) and run their tests.

    dbt owns the gold definitions; batch, stream and Airflow all call this.
    """
    from dbt.cli.main import dbtRunner

    res = dbtRunner().invoke(["--quiet", "build", "--select", "tag:gold",
                              "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)])
    if not res.success:
        raise GoldBuildError(f"dbt build failed: {res.exception or res.result}")
    out = {m: t.scalar(f"SELECT count(*) FROM lakehouse.gold.{m}") for m in GOLD_MODELS}
    print(f"gold (dbt): {out}")
    return out
