"""Shared clients and helpers for the lakehouse pipeline (S3 landing zone + Trino).

Endpoints default to the host view (Windows -> WSL port forwards). Inside Docker
(Airflow) set S3_ENDPOINT=http://seaweedfs:8333 and TRINO_HOST=trino.
"""

import io
import os
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import trino

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(l.split("=", 1) for l in (ROOT / "platform" / ".env").read_text(encoding="utf-8").splitlines() if "=" in l)

RAW_BUCKET = "raw"
LANDING = "landing"


S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
TRINO_HOST = os.environ.get("TRINO_HOST", "localhost")
TRINO_PORT = int(os.environ.get("TRINO_PORT", "8080"))


def s3():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT, region_name="us-east-1",
                        aws_access_key_id=ENV["S3_ACCESS_KEY"], aws_secret_access_key=ENV["S3_SECRET_KEY"])


def put_parquet(client, table: pa.Table, key: str) -> None:
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    client.put_object(Bucket=RAW_BUCKET, Key=key, Body=buf.getvalue())


def clear_prefix(client, prefix: str) -> None:
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=RAW_BUCKET, Prefix=prefix)
            for o in page.get("Contents", [])]
    for i in range(0, len(keys), 1000):
        client.delete_objects(Bucket=RAW_BUCKET, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]})


class Trino:
    def __init__(self):
        self.conn = trino.dbapi.connect(host=TRINO_HOST, port=TRINO_PORT, user="pipeline")

    def run(self, sql: str) -> list:
        cur = self.conn.cursor()
        cur.execute(sql)
        try:
            return cur.fetchall()
        except trino.exceptions.TrinoUserError:
            raise
        except Exception:  # statements without a result set
            return []

    def scalar(self, sql: str):
        return self.run(sql)[0][0]


ARROW_TO_TRINO = {
    pa.int8(): "tinyint", pa.int16(): "smallint", pa.int32(): "integer", pa.int64(): "bigint",
    pa.float32(): "real", pa.float64(): "double", pa.bool_(): "boolean", pa.string(): "varchar",
    pa.large_string(): "varchar",
}


def trino_columns(schema: pa.Schema) -> list[tuple[str, str]]:
    return [(f.name, ARROW_TO_TRINO[f.type]) for f in schema]
