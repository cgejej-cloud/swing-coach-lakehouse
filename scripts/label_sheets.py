"""Render frame strips around each predicted event for visual labelling.

One image per video: 4 rows (P1, P4, P7, P10), frames around the prediction,
cropped to the golfer (feet and ball included). Frame numbers are printed on
every tile so the labeller can read off the event frame.
Usage: uv run --no-sync python scripts/label_sheets.py [--seq 123 ...]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_coach.video import frames, probe  # noqa: E402
from swing_coach.viz import imwrite  # noqa: E402

# (offsets from the prediction) per event: impact needs the finest steps.
OFFSETS = {
    "P1": range(-20, 21, 5),
    "P4": range(-20, 21, 5),
    "P7": range(-8, 9, 2),
    "P10": range(-20, 21, 5),
}


def load_items() -> list[dict]:
    html = (ROOT / "label" / "labeler.html").read_text(encoding="utf-8")
    return json.loads(re.search(r"const ITEMS = (\[.*?\]);", html, re.S).group(1))


def sheet(item: dict, offsets: dict[str, range] = OFFSETS, centers: dict[str, int] | None = None) -> np.ndarray:
    seq = item["seq"]
    video = ROOT / "nasmo" / "videos" / f"{seq}.mp4"
    info = probe(video)
    centers = centers or item["pred"]
    pose = pd.read_parquet(ROOT / "data" / "pose" / f"{seq}.parquet")
    xs, ys = pose.x * info.width, pose.y * info.height
    x0, x1 = int(max(0, xs.quantile(0.01) - 120)), int(min(info.width, xs.quantile(0.99) + 120))
    y0, y1 = int(max(0, ys.quantile(0.01) - 160)), int(min(info.height, ys.quantile(0.99) + 60))

    wanted = {}
    for ev, offs in offsets.items():
        for o in offs:
            f = int(np.clip(centers[ev] + o, 0, info.n_frames - 1))
            wanted.setdefault(f, []).append((ev, o))
    tiles = {}
    for i, rgb in enumerate(frames(video, info)):
        if i in wanted:
            t = cv2.resize(rgb[y0:y1, x0:x1], (150, int(150 * (y1 - y0) / (x1 - x0))))
            for ev, o in wanted[i]:
                tile = t.copy()
                cv2.rectangle(tile, (0, 0), (150, 22), (0, 0, 0), -1)
                cv2.putText(tile, f"{ev} f{i}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0) if o == 0 else (255, 255, 255), 1, cv2.LINE_AA)
                tiles[(ev, o)] = tile
    rows = [np.hstack([tiles[(ev, o)] for o in offs]) for ev, offs in offsets.items()]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*", type=int)
    args = ap.parse_args()
    out = ROOT / "data" / "label_sheets"
    for item in load_items():
        if args.seq and item["seq"] not in args.seq:
            continue
        imwrite(out / f"{item['seq']}.jpg", sheet(item))
        print(item["seq"], item["club"], item["pred"])


if __name__ == "__main__":
    main()
