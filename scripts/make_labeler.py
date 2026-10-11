"""Build label/labeler.html: a local page for marking P1/P4/P7/P10 frames by hand.

Picks a stratified sample (club x ball flight) of HD videos, pre-fills the
detector's guesses, and lets the user step frame by frame to correct them.
Usage: uv run python scripts/make_labeler.py [--n 30]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "labeler_template.html"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "swing_metrics.csv")
    df = df[(df.width == 1280) & df.p4.notna()]
    per_club = args.n // 2
    picks = []
    for club, g in df.groupby("club"):
        # Spread picks across ball-flight types, then fill with the rest.
        by_flight = [x.sample(frac=1, random_state=7) for _, x in g.groupby("ball_flight_code")]
        order = pd.concat([pd.concat([b.iloc[[i]] for b in by_flight if i < len(b)])
                           for i in range(max(map(len, by_flight)))])
        picks.append(order.head(per_club))
    sample = pd.concat(picks)

    items = [{
        "seq": int(r.nasmo_seq), "club": r.club, "flight": int(r.ball_flight_code), "fps": float(r.fps),
        "src": f"../nasmo/videos/{int(r.nasmo_seq)}.mp4",
        "pred": {"P1": int(r.p1), "P4": int(r.p4), "P7": int(r.p7), "P10": int(r.p10)},
    } for r in sample.itertuples()]

    out = ROOT / "label" / "labeler.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("/*ITEMS*/[]", json.dumps(items)), encoding="utf-8")
    print(f"{len(items)} videos -> {out}")


if __name__ == "__main__":
    main()
