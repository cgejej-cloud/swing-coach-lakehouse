"""Extract golfer poses for nasmo videos into data/pose/{nasmo_seq}.parquet.

Usage: uv run python scripts/extract_poses.py [--club DRIVER IRON] [--limit N] [--seq 123 456]
Already-extracted videos are skipped, so the script can be re-run safely.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The editable-install .pth is ignored when the project path contains Korean characters.
sys.path.insert(0, str(ROOT / "src"))

from swing_coach.pose import extract  # noqa: E402

VIDEOS = ROOT / "nasmo" / "videos"
META = ROOT / "nasmo" / "nasmo_all_201.csv"
MODEL = ROOT / "models" / "pose_landmarker_full.task"
OUT = ROOT / "data" / "pose"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--club", nargs="*", default=["DRIVER", "IRON"])
    ap.add_argument("--seq", nargs="*", type=int)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    meta = pd.read_csv(META, encoding="utf-8-sig")
    todo = meta[meta.club.isin(args.club)] if not args.seq else meta[meta.nasmo_seq.isin(args.seq)]
    if args.limit:
        todo = todo.head(args.limit)
    OUT.mkdir(parents=True, exist_ok=True)

    for seq in todo.nasmo_seq:
        out = OUT / f"{seq}.parquet"
        if out.exists():
            continue
        t0 = time.perf_counter()
        df, info = extract(VIDEOS / f"{seq}.mp4", MODEL)
        df.insert(0, "nasmo_seq", seq)
        df.attrs.update(width=info.width, height=info.height, fps=info.fps, n_frames=info.n_frames)
        df.to_parquet(out, index=False)
        cover = df.frame.nunique() / info.n_frames
        print(f"{seq}: {info.width}x{info.height}@{info.fps:.0f} frames={info.n_frames} "
              f"detected={cover:.0%} vis={df.vis.mean():.2f} {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
