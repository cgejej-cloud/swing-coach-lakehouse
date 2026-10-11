"""Run phase detection + metrics on extracted poses; render P1/P4/P7/P10 rows for a visual check.

Usage: uv run python scripts/check_phases.py [--club DRIVER] [--sheet 8]
Writes data/swing_metrics.csv and data/viz/phases_{club}.jpg (first N videos).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_coach.metrics import compute, fix_sides  # noqa: E402
from swing_coach.phases import PhaseDetectionError, detect  # noqa: E402
from swing_coach.video import frames, probe  # noqa: E402
from swing_coach.viz import contact_sheet, draw_skeleton, imwrite  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--club", nargs="*", default=["DRIVER", "IRON"])
    ap.add_argument("--sheet", type=int, default=8)
    args = ap.parse_args()

    meta = pd.read_csv(ROOT / "nasmo" / "nasmo_all_201.csv", encoding="utf-8-sig")
    meta = meta[meta.club.isin(args.club)]
    rows, sheet_imgs, sheet_labels = [], [], []
    for m in meta.itertuples():
        path = ROOT / "data" / "pose" / f"{m.nasmo_seq}.parquet"
        if not path.exists():
            continue
        pose = pd.read_parquet(path)
        info = probe(ROOT / "nasmo" / "videos" / f"{m.nasmo_seq}.mp4")
        base = {"nasmo_seq": m.nasmo_seq, "club": m.club, "ball_flight_code": m.ball_flight_code,
                "carry_m": m.carry_m, "fps": info.fps, "width": info.width}
        try:
            ph = detect(pose, info.fps, info.n_frames)
        except PhaseDetectionError as e:
            rows.append(base | {"error": str(e)})
            continue
        rows.append(base | {"p1": ph.p1, "p4": ph.p4, "p7": ph.p7, "p10": ph.p10} | compute(pose, ph))

        if len(sheet_imgs) < args.sheet * 4:
            fixed = fix_sides(pose, ph.p1)
            wanted = {ph.p1: "P1", ph.p4: "P4", ph.p7: "P7", ph.p10: "P10"}
            for i, rgb in enumerate(frames(ROOT / "nasmo" / "videos" / f"{m.nasmo_seq}.mp4", info)):
                if i in wanted:
                    img = draw_skeleton(rgb.copy(), fixed[fixed.frame == i])
                    sheet_imgs.append(img)
                    sheet_labels.append(f"{m.nasmo_seq % 10000} {wanted[i]} f{i} t{ph.tempo:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "swing_metrics.csv", index=False)
    if sheet_imgs:
        imwrite(ROOT / "data" / "viz" / f"phases_{'_'.join(args.club)}.jpg",
                contact_sheet(sheet_imgs, sheet_labels, cols=4, width=300))
    ok = df["error"].isna() if "error" in df else pd.Series(True, index=df.index)
    print(f"videos={len(df)} detected={ok.sum()} failed={(~ok).sum()}")
    if ok.any():
        print(df[ok][["tempo_ratio", "backswing_s", "downswing_s", "phase_confidence"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
