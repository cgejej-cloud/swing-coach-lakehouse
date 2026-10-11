"""Execution accuracy of the text-to-SQL agent on tests/ask_eval.json.

A question counts as correct when every value of the reference result appears in the
agent's result (numbers compared after rounding to 2 decimals), with the same row count.
Extra columns in the agent's answer are allowed.
Usage: uv run --no-sync python scripts/eval_ask.py
"""

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_coach.ask import ask, trino_runner  # noqa: E402


def norm(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{round(float(v), 2):.2f}"
    if isinstance(v, (datetime, date)):
        return v.isoformat()[:10]
    return str(v).strip().lower()


def correct(ref_rows, got_rows) -> bool:
    if len(ref_rows) != len(got_rows):
        return False
    got = {norm(v) for row in got_rows for v in row}
    return all(norm(v) in got for row in ref_rows for v in row)


def run_once(run, cases, verbose: bool) -> int:
    ok = 0
    for c in cases:
        _, ref = run(c["sql"])
        state = ask(c["q"], run)
        good = "rows" in state and correct(ref, state["rows"])
        ok += good
        if verbose:
            print(f"[{'OK ' if good else 'BAD'}] {c['q']}\n      -> {state.get('answer', '')[:110]}"
                  + ("" if good else f"\n      sql: {state.get('sql', '')[:200]}"))
    return ok


def main() -> None:
    import argparse
    import statistics

    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3, help="runs; LLM output varies run to run")
    args = ap.parse_args()
    run = trino_runner()
    cases = json.loads((ROOT / "tests" / "ask_eval.json").read_text(encoding="utf-8"))
    t0 = time.time()
    scores = [run_once(run, cases, verbose=(i == 0)) for i in range(args.repeat)]
    n = len(cases)
    print(f"\nexecution accuracy over {args.repeat} runs: {[f'{s}/{n}' for s in scores]} "
          f"mean {statistics.mean(scores) / n:.0%} (min {min(scores) / n:.0%}, max {max(scores) / n:.0%}), "
          f"{(time.time() - t0) / (n * args.repeat):.1f}s/question")


if __name__ == "__main__":
    main()
