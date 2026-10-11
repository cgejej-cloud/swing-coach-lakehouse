"""Grid-search the top / finish parameters of phases.py against the hand labels.

Usage: uv run --no-sync python scripts/tune_phases.py
"""

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import swing_coach.phases as P  # noqa: E402
from eval_phases import evaluate  # noqa: E402

results = []
for smooth, band in itertools.product([0.05, 0.1, 0.2, 0.35], [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]):
    P.TOP_SMOOTH_S, P.TOP_BAND = smooth, band
    df = evaluate(verbose=False)
    results.append((df.P4_err.abs().mean(), df.P4_err.mean(), (df.P4_err.abs() <= 3).mean(), smooth, band,
                    (df.tempo_pred - df.tempo_label).abs().mean()))
print("top: MAE, bias, within3, smooth_s, band, tempo MAE")
for r in sorted(results)[:6]:
    print([round(v, 3) for v in r])

best_top = sorted(results)[0]
P.TOP_SMOOTH_S, P.TOP_BAND = best_top[3], best_top[4]
fin = []
for band in [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]:
    P.FINISH_BAND = band
    df = evaluate(verbose=False)
    fin.append((df.P10_err.abs().mean(), df.P10_err.mean(), (df.P10_err.abs() <= 3).mean(), band))
print("finish: MAE, bias, within3, band")
for r in sorted(fin)[:5]:
    print([round(v, 3) for v in r])
