"""One-time move of the app DB from SQLite (data/app.db) to Postgres (APP_DATABASE_URL).

Keeps primary keys, resets the identity sequence, then verifies row counts.
Refuses to run if the Postgres tables already have rows.
Usage: uv run --no-sync python platform/migrate_app_db.py
"""

import sqlite3
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swing_coach.store import Store  # noqa: E402

ENV = dict(l.split("=", 1) for l in (ROOT / "platform" / ".env").read_text(encoding="utf-8").splitlines() if "=" in l)
URL = ENV["APP_DATABASE_URL"]
TABLES = ["videos", "swings", "cards", "reference"]  # parent first (foreign keys)


def ensure_database() -> None:
    """App DB plus Airflow's metadata DB, both on the platform Postgres."""
    admin = URL.rsplit("/", 1)[0] + "/lakekeeper"
    with psycopg.connect(admin, autocommit=True) as c:
        for db in ("swingcoach", "airflow"):
            if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,)).fetchone():
                c.execute(f"CREATE DATABASE {db}")
                print(f"created database {db}")


def main() -> None:
    ensure_database()
    Store(URL)  # creates the schema
    src = sqlite3.connect(ROOT / "data" / "app.db")
    src.row_factory = sqlite3.Row
    with psycopg.connect(URL) as dst:
        if dst.execute("SELECT count(*) FROM videos").fetchone()[0]:
            sys.exit("Postgres already has videos; not migrating twice.")
        for t in TABLES:
            rows = src.execute(f"SELECT * FROM {t}").fetchall()
            if not rows:
                continue
            cols = rows[0].keys()
            with dst.cursor() as cur:
                cur.executemany(f"INSERT INTO {t} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})",
                                [tuple(r) for r in rows])
            print(f"{t}: {len(rows)} rows")
        dst.execute("SELECT setval(pg_get_serial_sequence('videos', 'id'), (SELECT max(id) FROM videos))")
        for t in TABLES:
            a = src.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            b = dst.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            assert a == b, f"{t}: sqlite={a} postgres={b}"
    print("migrated and verified")


if __name__ == "__main__":
    main()
