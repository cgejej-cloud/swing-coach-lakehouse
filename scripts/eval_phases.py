"""Compare phase detection with the hand labels in label/phase_labels.csv.

Labels were read off frame strips (5-frame steps, 2-frame for impact and for
re-checked clips), so they carry roughly +-2-3 frames of uncertainty themselves.
Usage: uv run --no-sync python scripts/eval_phases.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_coach.phases import PhaseDetectionError, detect  # noqa: E402
from swing_coach.video import probe  # noqa: E402

EVENTS = ["P1", "P4", "P7", "P10"]


def evaluate(detect_fn=detect, verbose: bool = True) -> pd.DataFrame:
    labels = pd.read_csv(ROOT / "label" / "phase_labels.csv")
    rows = []
    for r in labels.itertuples():
        pose = pd.read_parquet(ROOT / "data" / "pose" / f"{r.nasmo_seq}.parquet")
        info = probe(ROOT / "nasmo" / "videos" / f"{r.nasmo_seq}.mp4")
        try:
            ph = detect_fn(pose, info.fps, info.n_frames)
            pred = {"P1": ph.p1, "P4": ph.p4, "P7": ph.p7, "P10": ph.p10}
        except PhaseDetectionError:
            pred = {e: np.nan for e in EVENTS}
        rows.append({"nasmo_seq": r.nasmo_seq, **{f"{e}_label": getattr(r, e) for e in EVENTS},
                     **{f"{e}_pred": pred[e] for e in EVENTS}})
    df = pd.DataFrame(rows)
    for e in EVENTS:
        df[f"{e}_err"] = df[f"{e}_pred"] - df[f"{e}_label"]
    df["tempo_label"] = (df.P4_label - df.P1_label) / (df.P7_label - df.P4_label)
    df["tempo_pred"] = (df.P4_pred - df.P1_pred) / (df.P7_pred - df.P4_pred)
    if verbose:
        summary = pd.DataFrame({e: {
            "MAE(frames)": df[f"{e}_err"].abs().mean(),
            "bias(frames)": df[f"{e}_err"].mean(),
            "within ±3": (df[f"{e}_err"].abs() <= 3).mean(),
            "within ±6": (df[f"{e}_err"].abs() <= 6).mean(),
        } for e in EVENTS}).T
        print(summary.round(2).to_string())
        terr = (df.tempo_pred - df.tempo_label).abs()
        print(f"tempo: label median {df.tempo_label.median():.2f}, pred median {df.tempo_pred.median():.2f}, "
              f"MAE {terr.mean():.2f}, within ±0.5: {(terr <= 0.5).mean():.0%}")
    return df


if __name__ == "__main__":
    evaluate()
