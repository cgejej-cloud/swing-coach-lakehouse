"""One-time setup: S3 buckets in SeaweedFS, Lakekeeper bootstrap, and the 'golf' warehouse.

Idempotent: existing buckets / warehouse are left alone.
Usage: uv run --no-sync python platform/bootstrap.py
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
ENV = dict(l.split("=", 1) for l in (HERE / ".env").read_text(encoding="utf-8").splitlines() if "=" in l)
S3_LOCAL = "http://localhost:8333"
LAKEKEEPER = "http://localhost:8181"
BUCKETS = ["raw", "lakehouse"]


def s3():
    return boto3.client("s3", endpoint_url=S3_LOCAL, region_name="us-east-1",
                        aws_access_key_id=ENV["S3_ACCESS_KEY"], aws_secret_access_key=ENV["S3_SECRET_KEY"])


def post(path: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(LAKEKEEPER + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main() -> None:
    client = s3()
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    for b in BUCKETS:
        if b not in existing:
            client.create_bucket(Bucket=b)
    print("buckets:", sorted({b["Name"] for b in client.list_buckets()["Buckets"]}))

    print("bootstrap:", post("/management/v1/bootstrap", {"accept-terms-of-use": True})[0])
    status, body = post("/management/v1/warehouse", {
        "warehouse-name": "golf",
        "storage-profile": {
            "type": "s3", "bucket": "lakehouse", "key-prefix": "warehouse", "region": "us-east-1",
            # Endpoint as seen from inside the compose network (Trino, Lakekeeper).
            "endpoint": "http://seaweedfs:8333", "path-style-access": True,
            "flavor": "s3-compat", "sts-enabled": False,
        },
        "storage-credential": {
            "type": "s3", "credential-type": "access-key",
            "aws-access-key-id": ENV["S3_ACCESS_KEY"], "aws-secret-access-key": ENV["S3_SECRET_KEY"],
        },
    })
    print("warehouse:", status, body[:300])


if __name__ == "__main__":
    main()
