"""Create local-only credentials for the lakehouse stack (platform/.env, seaweedfs/s3.json).

Existing files are kept, so re-running never rotates keys under running services.
Both files are git-ignored.
"""

import json
import secrets
from pathlib import Path

HERE = Path(__file__).resolve().parent
env_path = HERE / ".env"
s3_path = HERE / "seaweedfs" / "s3.json"

if not env_path.exists():
    values = {
        "PG_USER": "lakekeeper",
        "PG_PASSWORD": secrets.token_urlsafe(18),
        "LAKEKEEPER_ENCRYPTION_KEY": secrets.token_urlsafe(32),
        "S3_ACCESS_KEY": "swingcoach",
        "S3_SECRET_KEY": secrets.token_urlsafe(24),
    }
    env_path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8", newline="\n")

env = dict(line.split("=", 1) for line in env_path.read_text(encoding="utf-8").splitlines() if "=" in line)
if "APP_DATABASE_URL" not in env:
    # App DB lives in the same Postgres server (database "swingcoach"), reached from Windows on localhost.
    env["APP_DATABASE_URL"] = f"postgresql://{env['PG_USER']}:{env['PG_PASSWORD']}@localhost:5432/swingcoach"
    with env_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"APP_DATABASE_URL={env['APP_DATABASE_URL']}\n")
if not s3_path.exists():
    s3_path.parent.mkdir(exist_ok=True)
    s3_path.write_text(json.dumps({"identities": [{
        "name": "swingcoach",
        "credentials": [{"accessKey": env["S3_ACCESS_KEY"], "secretKey": env["S3_SECRET_KEY"]}],
        "actions": ["Admin", "Read", "Write", "List", "Tagging"],
    }]}, indent=2), encoding="utf-8", newline="\n")
print("ok:", env_path.name, s3_path.relative_to(HERE))
