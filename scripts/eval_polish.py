"""Run the LLM card polisher on real cards: acceptance rate, rejection reasons, examples.

Usage: uv run --no-sync python scripts/eval_polish.py [--n 30]
"""

import argparse
import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_coach import llm  # noqa: E402
from swing_coach.coach import Card  # noqa: E402
from swing_coach.store import default_store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    with default_store().conn() as c:
        rows = c.execute("SELECT video_id, good, fix, cue FROM cards WHERE fix IS NOT NULL "
                         "ORDER BY video_id DESC LIMIT ?", (args.n,)).fetchall()
    reasons, examples, t0 = collections.Counter(), [], time.time()
    for r in rows:
        card = Card(good=r["good"], fix=r["fix"], cue=r["cue"])
        applied, note = llm.polish_card(card)
        reasons[note if not applied else "polished"] += 1
        if applied and len(examples) < 4:
            examples.append((r, card))
    n = len(rows)
    print(f"cards={n} accepted={reasons['polished']} ({reasons['polished'] / n:.0%}) "
          f"avg {(time.time() - t0) / n:.1f}s/card")
    for k, v in reasons.most_common():
        print(f"  {v:3d}  {k}")
    for r, card in examples:
        print("\n원문:", r["good"], "|", r["fix"], "|", r["cue"])
        print("수정:", card.good, "|", card.fix, "|", card.cue)


if __name__ == "__main__":
    main()
