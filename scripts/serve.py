"""Run the Swing Coach app on http://localhost:8000 (add --host 0.0.0.0 to reach it from a phone on the same Wi-Fi)."""

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    from swing_coach.app import app  # noqa: E402

    uvicorn.run(app, host=args.host, port=args.port)
