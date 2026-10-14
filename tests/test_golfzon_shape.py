"""Shaping of Golfzon round records (pipeline/golfzon.py) on a synthetic round."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import golfzon  # noqa: E402

PAR = [4] * 18


def _round(holes: dict) -> dict:
    return {"pg": 1, "title": "테스트 CC 80타 (+8점)", "played_at": "2026-10-01 10:00:00", "course_parts": ["A", "B"],
            "par": PAR, "strokes": [4] * 18, "putts": [2] * 18, "penalty": [0] * 18,
            "tee_flight_class": ["txt_draw"] * 9 + ["td_sum"] + ["txt_slice"] * 9 + ["td_sum", "td_sum"],
            "driver": {"driverDistanceAvg": 210.0, "driverDistanceMax": 240.0},
            "page_tables": ["구분 이번경기 개인평균 정타확율 5.8% 7.2% 비거리 222m 198.7m"],
            "holes": holes, "hole_length_m": {str(h): 350 for h in range(1, 19)}}


def _shots(holes):
    rnd, hole_rows, shots = golfzon.shape_round(_round(holes))
    return rnd, hole_rows, shots


def test_par_four_in_regulation():
    rows = [[230, 120, 1, 0, 0], [118, 5, 3, 0, 0], [5, 0.5, 3, 0, 0], [0.5, 0, 8, 0, 0]]
    _, holes, shots = _shots({"1": rows})
    h = holes[0]
    assert (h["gir"], h["fairway_hit"], h["swings_to_green"], h["tee_flight"]) == (1, 1, 2, "드로우")
    assert [s["start_remain_m"] for s in shots] == [350, 120, 5, 0.5]
    assert [s["start_on_green"] for s in shots] == [0, 0, 1, 1]
    assert holes[9]["tee_flight"] == "슬라이스"   # the 'td_sum' cells between nines are skipped


def test_penalty_area_then_drop_zone_is_a_relocation():
    rows = [[200, 150, 6, 0, 0], [20, 170, 27, 0, 0], [160, 10, 3, 0, 0], [10, 0, 8, 0, 0]]
    _, holes, shots = _shots({"1": rows})
    assert [s["is_relocation"] for s in shots] == [0, 1, 0, 0]
    assert [s["swing_no"] for s in shots] == [1, None, 2, 3]
    assert shots[0]["is_penalty_lie"] == 1
    assert shots[2]["start_remain_m"] == 170   # the next swing starts from the drop zone
    assert holes[0]["relocations"] == 1 and holes[0]["gir"] == 1


def test_ob_followed_by_replay_from_the_same_spot_is_a_relocation():
    # Some courses re-list the previous spot (a normal lie code) instead of a front tee.
    rows = [[200, 150, 1, 0, 0], [130, 170, 9, 0, 0], [130, 150, 1, 0, 0], [140, 10, 3, 0, 0], [10, 0, 8, 0, 0]]
    _, holes, shots = _shots({"1": rows})
    assert [s["is_relocation"] for s in shots] == [0, 0, 1, 0, 0]
    assert holes[0]["relocations"] == 1


def test_round_totals_and_shot_table_values():
    rnd, _, _ = _shots({})
    assert rnd["course"] == "테스트 CC" and rnd["strokes"] == 72 and rnd["putts"] == 36
    assert rnd["sweet_spot_pct"] == 5.8 and rnd["driver_carry_m"] == 222


def test_driver_landings_are_deduplicated_across_round_files(tmp_path):
    a = _round({}) | {"pg": 1, "driver_landing": [{"pg": 1, "T": [[1, -5.0, 220.0, 0]]}]}
    b = _round({}) | {"pg": 2, "driver_landing": [{"pg": 1, "T": [[1, -5.0, 220.0, 0]]},
                                                  {"pg": 2, "T": [[1, 12.0, 230.0, 1]]}]}
    for r in (a, b):
        (tmp_path / f"{r['pg']}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    t = golfzon.tables(tmp_path)
    assert len(t["golfzon_driver_landing"]) == 2
    assert len(t["golfzon_hole"]) == 36 and list(t["golfzon_round"].columns) == golfzon.COLUMNS["golfzon_round"]


def test_no_round_files_still_gives_the_table_schema(tmp_path):
    t = golfzon.tables(tmp_path)
    assert all(df.empty and list(df.columns) == golfzon.COLUMNS[name] for name, df in t.items())
