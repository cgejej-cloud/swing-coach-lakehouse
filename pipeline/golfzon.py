"""Golfzon round files -> landing tables: rounds, holes, every shot, driver landings (ADR 007).

Input: data/golfzon/rounds/{pg_code}.json, one per round, kept exactly as collected so the
tables can be rebuilt when this shaping changes. Holes are numbered 1-18 in play order.

How the shot list reads (learned from 10 rounds, see tests/test_golfzon_shape.py):
  * a row is where the ball came to rest: (distance, remain to the hole, lie code, x, y)
  * penalties appear as extra rows that move the ball without a swing: a drop-zone row
    after a penalty area, a front-tee row after OB, or (some courses) a row back at the
    previous spot after OB. Those are `is_relocation`; they match the scorecard penalties.
  * the final putt is sometimes missing, so hole strokes come from the scorecard, and the
    shot list is checked against it as a warning, not an error.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROUNDS_DIR = ROOT / "data" / "golfzon" / "rounds"

# Lie codes seen on the hole-detail pages; each file also carries the labels it saw.
LIES = {1: "페어웨이", 2: "러프", 3: "그린", 4: "벙커", 5: "그린벙커", 6: "패널티 에어리어", 8: "홀인",
        9: "OB", 10: "에이프런", 13: "프런트티", 23: "헤비러프", 27: "드롭존"}
PENALTY_LIES = {6, 9}
RELOCATION_LIES = {13, 27}
GREEN_LIES = {3, 8}
FLIGHT = {"txt_straight": "스트레이트", "txt_draw": "드로우", "txt_fade": "페이드", "txt_hook": "훅",
          "txt_slice": "슬라이스", "txt_push": "푸시", "txt_pull": "풀"}
SHOT_METRICS = {"정타확율": "sweet_spot_pct", "비거리": "driver_carry_m", "발사각": "launch_deg",
                "좌우편차": "lateral_dev_m", "볼스피드": "ball_speed_ms", "좌우스핀": "side_spin_rpm"}

INT_COLUMNS = {
    "golfzon_round": ["pg_code", "par", "strokes", "putts", "penalties", "holes"],
    "golfzon_hole": ["pg_code", "hole_no", "hole_in_part", "par", "strokes", "putts", "penalties", "score_to_par",
                     "hole_length_m", "tee_lie_code", "swings_to_green", "gir", "fairway_hit", "shot_rows",
                     "relocations"],
    "golfzon_shot": ["pg_code", "hole_no", "shot_no", "swing_no", "lie_code", "start_lie_code", "is_penalty_lie",
                     "is_relocation", "start_on_green"],
    "golfzon_driver_landing": ["pg_code", "hole_no", "attr"],
}
FLOAT_COLUMNS = {
    "golfzon_round": ["driver_avg_m", "driver_max_m"] + list(SHOT_METRICS.values()),
    "golfzon_shot": ["distance_m", "start_remain_m", "end_remain_m", "x", "y"],
    "golfzon_driver_landing": ["x_m", "y_m"],
}


# Column order of each landing table; also the schema when there are no round files yet.
COLUMNS = {
    "golfzon_round": ["pg_code", "course", "played_at", "par", "strokes", "putts", "penalties", "holes",
                      "driver_avg_m", "driver_max_m", *SHOT_METRICS.values()],
    "golfzon_hole": ["pg_code", "hole_no", "course_part", "hole_in_part", "par", "hole_length_m", "strokes", "putts",
                     "penalties", "score_to_par", "tee_lie_code", "tee_lie", "tee_flight", "swings_to_green", "gir",
                     "fairway_hit", "shot_rows", "relocations"],
    "golfzon_shot": ["pg_code", "hole_no", "shot_no", "swing_no", "start_lie_code", "start_lie", "start_remain_m",
                     "distance_m", "end_remain_m", "lie_code", "lie", "is_penalty_lie", "is_relocation",
                     "start_on_green", "x", "y"],
    "golfzon_driver_landing": ["pg_code", "hole_no", "x_m", "y_m", "attr"],
}


def _shot_table_values(page_tables: list[str]) -> dict[str, float]:
    """'정타확율 5.8% 7.2% ...' -> this round's value (the first number after each label)."""
    text = " ".join(page_tables)
    out = {}
    for label, col in SHOT_METRICS.items():
        m = re.search(label + r"\s+(-?[\d.]+)", text)
        out[col] = float(m.group(1)) if m else None
    return out


def shape_round(r: dict) -> tuple[dict, list[dict], list[dict]]:
    pg = int(r["pg"])
    lies = LIES | {int(k): v for k, v in (r.get("lies") or {}).items()}
    flights = [FLIGHT.get(c) for c in r.get("tee_flight_class", []) if c != "td_sum"]
    parts = r.get("course_parts") or []
    holes, shots = [], []
    for h in range(1, len(r["par"]) + 1):
        par, rows = r["par"][h - 1], r["holes"].get(str(h), [])
        length = (r.get("hole_length_m") or {}).get(str(h))
        prev_remain, prev_lie, swing_no, relocations, to_green = length, None, 0, 0, None
        for i, (dist, remain, lie, x, y) in enumerate(rows, start=1):
            relocation = lie in RELOCATION_LIES or prev_lie == 9
            relocations += relocation
            if not relocation:
                swing_no += 1
                if to_green is None and lie in GREEN_LIES:
                    to_green = swing_no
            shots.append({
                "pg_code": pg, "hole_no": h, "shot_no": i, "swing_no": None if relocation else swing_no,
                "start_lie_code": prev_lie, "start_lie": lies.get(prev_lie, "티") if prev_lie else "티",
                "start_remain_m": prev_remain, "distance_m": dist, "end_remain_m": remain,
                "lie_code": lie, "lie": lies.get(lie, f"코드{lie}"), "is_penalty_lie": int(lie in PENALTY_LIES),
                "is_relocation": int(relocation), "start_on_green": int(prev_lie in GREEN_LIES), "x": x, "y": y})
            prev_remain, prev_lie = remain, lie
        strokes = r["strokes"][h - 1]
        tee_lie = rows[0][2] if rows else None
        holes.append({
            "pg_code": pg, "hole_no": h, "course_part": parts[(h - 1) // 9] if len(parts) > (h - 1) // 9 else None,
            "hole_in_part": (h - 1) % 9 + 1, "par": par, "hole_length_m": length, "strokes": strokes,
            "putts": r["putts"][h - 1] or 0, "penalties": r["penalty"][h - 1] if r.get("penalty") else 0,
            "score_to_par": strokes - par, "tee_lie_code": tee_lie, "tee_lie": lies.get(tee_lie),
            "tee_flight": flights[h - 1] if len(flights) >= h else None,
            "swings_to_green": to_green, "gir": int(to_green is not None and to_green <= par - 2),
            "fairway_hit": int(tee_lie == 1) if par >= 4 else None, "shot_rows": len(rows), "relocations": relocations})
    title = r.get("title") or ""
    m = re.match(r"^(.*?)\s+(\d+)타", title)
    driver = r.get("driver") or {}
    rnd = {"pg_code": pg, "course": m.group(1) if m else title, "played_at": r.get("played_at"),
           "par": sum(r["par"]), "strokes": sum(r["strokes"]), "putts": sum(p or 0 for p in r["putts"]),
           "penalties": sum(r.get("penalty") or []), "holes": len(r["par"]),
           "driver_avg_m": driver.get("driverDistanceAvg"), "driver_max_m": driver.get("driverDistanceMax"),
           **_shot_table_values(r.get("page_tables") or [])}
    return rnd, holes, shots


def tables(rounds_dir: Path = ROUNDS_DIR) -> dict[str, pd.DataFrame]:
    rounds, holes, shots, landing = [], [], [], {}
    for path in sorted(rounds_dir.glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        rnd, h, s = shape_round(r)
        rounds.append(rnd)
        holes += h
        shots += s
        # Each page lists the last 10 rounds' driver landings, so rounds overlap: keep one per hole.
        for g in r.get("driver_landing") or []:
            for hole, x, y, attr in g["T"]:
                landing[(int(g["pg"]), int(hole))] = {"pg_code": int(g["pg"]), "hole_no": int(hole),
                                                      "x_m": x, "y_m": y, "attr": attr}
    rows = {"golfzon_round": rounds, "golfzon_hole": holes, "golfzon_shot": shots,
            "golfzon_driver_landing": list(landing.values())}
    return {name: pd.DataFrame(rows[name], columns=cols) for name, cols in COLUMNS.items()}
