"""Does the textbook verdict (rubric.py) line up with what the ball did?

For every analysed driver/iron shot with a Golfzon ball-flight label: verdict counts,
straight rate and median carry per verdict, and per core metric the straight rate when
it is inside vs outside the textbook band (Fisher exact test). Nothing is fitted to this
data, so there is no train/test split to worry about; with one golfer and ~170 shots,
read it as a sanity check, not proof.

Usage: uv run --no-sync python scripts/eval_rubric.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_coach import rubric  # noqa: E402
from swing_coach.store import default_store  # noqa: E402


def load() -> pd.DataFrame:
    with default_store().conn() as c:
        rows = c.execute("""SELECT v.id, v.club_group, v.ball_flight_code, v.carry_m, v.width,
                                   s.metrics_json, c.confidence
                            FROM swings s JOIN videos v ON v.id = s.video_id
                            LEFT JOIN cards c ON c.video_id = v.id""").fetchall()
    out = []
    for r in rows:
        m = json.loads(r["metrics_json"])
        v = rubric.judge(m, r["club_group"], r["confidence"] or "높음")
        rec = {"id": r["id"], "group": r["club_group"], "flight": r["ball_flight_code"],
               "carry": r["carry_m"], "verdict": v.level}
        rec.update({f"s_{c.key}": c.status for c in v.checks})
        out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    df = load()
    labelled = df[df.group.isin(["driver", "iron"]) & df.flight.isin([1, 2, 3])].copy()
    labelled["straight"] = labelled.flight == 1
    print(f"shots judged: {len(df[df.group.isin(['driver', 'iron'])])}, with ball-flight label: {len(labelled)}\n")
    for g, d in labelled.groupby("group"):
        print(f"== {g} (n={len(d)}, straight {d.straight.mean():.0%})")
        t = d.groupby("verdict").agg(n=("id", "size"), straight=("straight", "mean"), carry=("carry", "median"))
        print(t.round(2).to_string(), "\n")
        judged = d[d.verdict != "판정 보류"]
        good, poor = judged[judged.verdict == "잘함"], judged[judged.verdict == "아쉬움"]
        if len(good) and len(poor):
            p = fisher_exact([[good.straight.sum(), (~good.straight).sum()],
                              [poor.straight.sum(), (~poor.straight).sum()]])[1]
            pc = mannwhitneyu(good.carry.dropna(), poor.carry.dropna()).pvalue
            print(f"잘함 vs 아쉬움: straight {good.straight.mean():.0%} vs {poor.straight.mean():.0%} (Fisher p={p:.3f}), "
                  f"carry median {good.carry.median():.0f} vs {poor.carry.median():.0f}m (MWU p={pc:.3f})\n")
        print("core metric: share inside textbook band, straight rate inside vs outside")
        for rule in rubric.RULES:
            col = f"s_{rule.key}"
            if not rule.core[g] or col not in d:
                continue
            inside = d[d[col] == "green"]
            outside = d[d[col].isin(["amber", "red"])]
            if len(inside) and len(outside):
                p = fisher_exact([[inside.straight.sum(), (~inside.straight).sum()],
                                  [outside.straight.sum(), (~outside.straight).sum()]])[1]
                print(f"  {rule.key:30s} inside {len(inside) / len(d):4.0%}  "
                      f"straight {inside.straight.mean():.0%} vs {outside.straight.mean():.0%}  p={p:.3f}")
        print()


if __name__ == "__main__":
    main()
