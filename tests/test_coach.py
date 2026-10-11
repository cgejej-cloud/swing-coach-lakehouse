import numpy as np
import pandas as pd

from swing_coach.coach import CATALOG, make_card, robust_z

KEYS = [m.key for m in CATALOG]


def good_shots(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = {k: 0.0 for k in KEYS} | {"tempo_ratio": 3.5, "lead_knee_imp_deg": 175.0, "stance_width_sw": 1.1}
    return pd.DataFrame([{k: v + rng.normal(0, 0.05) for k, v in base.items()} for _ in range(n)])


def swing(**overrides) -> pd.Series:
    row = good_shots(1, seed=99).iloc[0].to_dict()
    row.update(phase_confidence=0.8, width=1280, scale_ok=1, ball_flight_code=1)
    row.update(overrides)
    return pd.Series(row)


def test_robust_z_ignores_outliers_in_reference():
    ref = pd.Series([1.0] * 20 + [100.0])
    assert abs(robust_z(1.0, ref)) < 1e-9


def test_fix_is_the_bad_direction_deviation():
    # Pelvis pushed far to the right at the top (more negative = sway): a fault.
    card = make_card(swing(pelvis_sway_top_sw=-0.8), good_shots(), None, {}, [])
    assert card.fix_key == "pelvis_sway_top_sw"
    assert "골반" in card.fix


def test_good_direction_deviation_is_not_a_fault():
    # Pelvis further toward the target than usual at impact is not "low" -> not flagged.
    card = make_card(swing(pelvis_shift_imp_sw=0.8), good_shots(), None, {}, [])
    assert card.fix_key != "pelvis_shift_imp_sw"


def test_low_confidence_withholds_advice():
    card = make_card(swing(pelvis_sway_top_sw=-0.8, phase_confidence=0.3), good_shots(), None, {}, [])
    assert card.fix is None and card.confidence == "낮음"


def test_flight_link_sentence_only_when_direction_matches():
    links = {"pelvis_sway_top_sw": [{"flight": 2, "sign": -1.0, "n": 50, "p": 0.004}]}
    sliced = make_card(swing(pelvis_sway_top_sw=-0.8, ball_flight_code=2), good_shots(), None, links, [])
    hooked = make_card(swing(pelvis_sway_top_sw=-0.8, ball_flight_code=3), good_shots(), None, links, [])
    assert any("슬라이스" in e for e in sliced.extra)
    assert not any("슬라이스" in e or "훅" in e for e in hooked.extra)


def test_third_repeat_suggests_a_drill():
    card = make_card(swing(pelvis_sway_top_sw=-0.8), good_shots(), None, {},
                     ["pelvis_sway_top_sw", "pelvis_sway_top_sw"])
    assert any("드릴" in e for e in card.extra)
