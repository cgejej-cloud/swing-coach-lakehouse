"""Register (or update) the Debezium Postgres source connector for the app tables.

Topics: swingcoach.public.{videos,swings,cards,reference}; JSON values without schema.
Usage: uv run --no-sync python platform/register_cdc.py
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = dict(l.split("=", 1) for l in (HERE / ".env").read_text(encoding="utf-8").splitlines() if "=" in l)
CONNECT = "http://localhost:8083"
NAME = "swingcoach-app"

CONFIG = {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": ENV["PG_USER"],
    "database.password": ENV["PG_PASSWORD"],
    "database.dbname": "swingcoach",
    "topic.prefix": "swingcoach",
    "plugin.name": "pgoutput",
    "slot.name": "swingcoach_slot",
    "publication.name": "swingcoach_pub",
    "publication.autocreate.mode": "filtered",
    "table.include.list": "public.videos,public.swings,public.cards,public.reference",
    "snapshot.mode": "initial",          # existing rows first (op = r), then the WAL stream
    "decimal.handling.mode": "double",
    "tombstones.on.delete": "false",
}


def request(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(CONNECT + path, json.dumps(body).encode() if body else None,
                                 {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or "null")


def main() -> None:
    for _ in range(60):
        try:
            request("GET", "/")
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(3)
    request("PUT", f"/connectors/{NAME}/config", CONFIG)  # create or update, idempotent
    status = {"connector": {"state": "UNKNOWN"}, "tasks": []}
    for _ in range(30):
        try:
            status = request("GET", f"/connectors/{NAME}/status")
        except urllib.error.HTTPError as e:  # 404 until the connector is created
            if e.code != 404:
                raise
            time.sleep(2)
            continue
        states = [status["connector"]["state"]] + [t["state"] for t in status.get("tasks", [])]
        if all(s == "RUNNING" for s in states) and len(states) > 1:
            break
        time.sleep(2)
    print(json.dumps({"connector": status["connector"]["state"], "tasks": [t["state"] for t in status["tasks"]]}))


if __name__ == "__main__":
    main()
