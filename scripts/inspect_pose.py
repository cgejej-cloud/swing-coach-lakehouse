"""Render a skeleton contact sheet and a wrist-height trace for one extracted video.

Usage: uv run python scripts/inspect_pose.py 5361377199 [--every 16]
Writes data/viz/{seq}_sheet.jpg and data/viz/{seq}_trace.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_coach.video import frames, probe  # noqa: E402
from swing_coach.viz import contact_sheet, draw_skeleton, imwrite  # noqa: E402

L_WRIST, R_WRIST, NOSE = 15, 16, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", type=int)
    ap.add_argument("--every", type=int, default=16)
    args = ap.parse_args()

    pose = pd.read_parquet(ROOT / "data" / "pose" / f"{args.seq}.parquet")
    video = ROOT / "nasmo" / "videos" / f"{args.seq}.mp4"
    info = probe(video)

    imgs, labels = [], []
    for i, rgb in enumerate(frames(video, info)):
        if i % args.every:
            continue
        img = rgb.copy()
        draw_skeleton(img, pose[pose.frame == i])
        imgs.append(img)
        labels.append(f"f{i} {i / info.fps:.2f}s")
    imwrite(ROOT / "data" / "viz" / f"{args.seq}_sheet.jpg", contact_sheet(imgs, labels))

    wide = pose[pose.lm.isin([L_WRIST, R_WRIST, NOSE])].pivot(index="frame", columns="lm", values=["x", "y"])
    trace = pd.DataFrame({
        "t": wide.index / info.fps,
        "hands_y": (wide[("y", L_WRIST)] + wide[("y", R_WRIST)]) / 2,
        "hands_x": (wide[("x", L_WRIST)] + wide[("x", R_WRIST)]) / 2,
        "nose_x": wide[("x", NOSE)],
    })
    trace.to_csv(ROOT / "data" / "viz" / f"{args.seq}_trace.csv")
    print(trace.iloc[::8].round(3).to_string())


if __name__ == "__main__":
    main()
