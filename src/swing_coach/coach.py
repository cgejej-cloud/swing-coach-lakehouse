"""Turn swing metrics into a short teaching card: one good point, one fix, one cue.

How a fix is chosen (MVP_SPEC 4.3):
  * every metric is compared with the golfer's own good shots (straight shots of
    the same club group) as a robust z-score - this is the "C" baseline;
  * only deviations in the metric's known "bad" direction count;
  * earlier swing phases win ties because they cause later faults;
  * the reference swing ("R" baseline) supplies the numbers quoted to the user.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Shoulder width assumed for turning shoulder-width units into ball counts
# in user-facing text. Only used for wording, never for decisions.
SHOULDER_CM = 40.0
BALL_CM = 4.27

# Tempo ("리듬") is weighted slightly down: against 30 hand-labelled clips the tempo
# ratio is off by 0.54 on average (about 15%), larger than the other metrics' noise.
PHASE_WEIGHT = {"셋업": 1.3, "백스윙": 1.2, "다운스윙": 1.0, "피니시": 0.8, "리듬": 0.8}


@dataclass(frozen=True)
class Metric:
    key: str
    name: str
    phase: str
    bad: str            # "low", "high" or "both"
    unit: str           # "sw", "deg", "ratio"
    fault_low: str = ""
    fault_high: str = ""
    cue_low: str = ""
    cue_high: str = ""
    drill: str = ""
    good_text: str = ""
    frame: str = "P4"   # key frame to highlight
    landmark: int = 0   # landmark to highlight on the card image


CATALOG = [
    Metric("stance_width_sw", "스탠스 폭", "셋업", "both", "sw",
           "스탠스가 평소 좋은 샷보다 좁아요", "스탠스가 평소 좋은 샷보다 넓어요",
           "발을 조금 더 벌려 서기", "발 간격을 조금 좁히기",
           "어드레스 전에 발 사이에 클럽을 놓고 폭 맞추기", "스탠스 폭이 안정적이에요", "P1", 27),
    Metric("trail_shoulder_drop_deg", "오른쪽 어깨 높이", "셋업", "both", "deg",
           "어드레스에서 오른쪽 어깨가 평소보다 높아요", "어드레스에서 오른쪽 어깨가 평소보다 많이 내려갔어요",
           "오른쪽 어깨를 살짝 낮춰 셋업", "어깨 높이를 조금 더 평평하게",
           "거울 앞에서 셋업 자세 사진 찍어 비교하기", "셋업 어깨 기울기가 좋아요", "P1", 12),
    Metric("head_sway_top_sw", "백스윙 머리 이동", "백스윙", "low", "sw",
           "탑에서 머리가 오른쪽으로 밀렸어요", "",
           "머리는 공 뒤에 두고 어깨만 돌리기", "",
           "머리 옆에 클럽을 세워 두고 닿지 않게 하프스윙 10번", "백스윙 때 머리가 잘 고정돼요", "P4", 0),
    Metric("pelvis_sway_top_sw", "백스윙 골반 이동", "백스윙", "low", "sw",
           "탑에서 골반이 오른쪽으로 밀렸어요", "",
           "오른 무릎을 버티고 골반은 제자리에서 돌리기", "",
           "오른발 바깥에 공을 두고 밀리지 않게 백스윙 연습", "백스윙 때 골반이 제자리에서 잘 돌아요", "P4", 24),
    Metric("tempo_ratio", "템포", "리듬", "both", "ratio",
           "백스윙이 평소보다 급해요", "탑에서 멈칫하거나 다운스윙이 평소보다 느려요",
           "백스윙은 천천히, 탑에서 한 박자", "탑에서 끊지 말고 이어서 내려오기",
           "'하나-둘-셋'에 올리고 '넷'에 치는 리듬 연습", "템포가 좋아요", "P4", 15),
    Metric("pelvis_shift_imp_sw", "임팩트 체중 이동", "다운스윙", "low", "sw",
           "임팩트 때 골반이 왼쪽으로 덜 넘어왔어요", "",
           "다운스윙은 왼쪽 골반부터 시작", "",
           "왼발 바깥에 클럽을 세우고 골반으로 살짝 닿기", "체중 이동이 좋아요", "P7", 23),
    Metric("head_move_imp_sw", "임팩트 머리 위치", "다운스윙", "high", "sw",
           "", "임팩트 때 머리가 공 쪽으로 따라 나갔어요",
           "", "임팩트까지 머리는 공 뒤에",
           "임팩트 후 공이 있던 자리를 1초 보고 고개 들기", "임팩트 때 머리가 공 뒤에 잘 남아요", "P7", 0),
    Metric("lead_knee_imp_deg", "임팩트 왼다리", "다운스윙", "low", "deg",
           "임팩트 때 왼무릎이 평소보다 굽어 있어요", "",
           "임팩트 때 왼다리로 버티기", "",
           "임팩트 자세에서 왼다리 펴고 3초 멈추기", "임팩트 때 왼다리가 잘 버텨요", "P7", 25),
    Metric("hands_ahead_imp_sw", "임팩트 손 위치", "다운스윙", "low", "sw",
           "임팩트 때 손이 평소보다 뒤에 있어요", "",
           "손이 공보다 먼저 지나가게", "",
           "하프스윙으로 손이 왼허벅지 앞에서 맞게 연습", "임팩트 손 위치가 좋아요", "P7", 15),
    Metric("pelvis_over_lead_foot_fin_sw", "피니시 체중", "피니시", "low", "sw",
           "피니시에서 체중이 오른쪽에 남았어요", "",
           "피니시는 왼발 위에 서기", "",
           "피니시 자세로 3초 버티기", "피니시 균형이 좋아요", "P10", 23),
]
BY_KEY = {m.key: m for m in CATALOG}
# Measured but not used for teaching: at the top the lead arm points toward a face-on
# camera, so its 2D elbow angle is foreshortened (driver median ~40 deg, iron ~116 deg).
UNRELIABLE_FACE_ON = ["lead_arm_top_deg", "lead_arm_fin_deg"]


@dataclass
class Card:
    good: str
    fix: str | None
    cue: str | None
    extra: list[str] = field(default_factory=list)
    fix_key: str | None = None
    good_key: str | None = None
    confidence: str = "높음"
    evidence: dict = field(default_factory=dict)


def robust_z(value: float, ref: pd.Series) -> float:
    med = ref.median()
    mad = (ref - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad < 1e-6:
        return 0.0
    return float((value - med) / mad)


def _magnitude(m: Metric, diff: float) -> str:
    if m.unit == "sw":
        balls = abs(diff) * SHOULDER_CM / BALL_CM
        return f"공 {max(1, round(balls))}개 정도"
    if m.unit == "deg":
        return f"{abs(diff):.0f}°"
    return f"{abs(diff):.1f}"


def _good_suffix(m: Metric, v: float) -> str:
    """Only tempo has a number people know; other metrics are relative to their own good shots."""
    return f" ({v:.1f}:1)" if m.unit == "ratio" else " (평소 좋은 샷과 같아요)"


def make_card(row: pd.Series, good_shots: pd.DataFrame, reference: pd.Series | None,
              flight_links: dict[str, dict], recent_fixes: list[str]) -> Card:
    """row: this swing's metrics; good_shots: straight shots of the same club group (self excluded)."""
    conf_score = row.get("phase_confidence", 0.0)
    low_res = row.get("width", 1280) < 640
    bad_scale = row.get("scale_ok", 1) == 0
    confidence = "낮음" if (conf_score < 0.45 or low_res or bad_scale) else ("보통" if conf_score < 0.6 else "높음")

    scored = []
    for m in CATALOG:
        if m.key not in row or pd.isna(row[m.key]) or len(good_shots) < 5:
            continue
        z = robust_z(row[m.key], good_shots[m.key].dropna())
        bad = (m.bad == "low" and z < 0) or (m.bad == "high" and z > 0) or m.bad == "both"
        scored.append((m, z, abs(z) * PHASE_WEIGHT[m.phase] if bad else 0.0))

    evidence = {m.key: {"value": float(row[m.key]), "z": round(z, 2),
                        "good_median": float(good_shots[m.key].median())} for m, z, _ in scored}

    if not scored:
        return Card(good="아직 비교할 좋은 샷이 부족해서 자세 표시만 했어요", fix=None, cue=None,
                    confidence=confidence, evidence=evidence)
    # Good point: the smallest deviation among metrics, preferring ones not chosen as fix.
    good_m, _, _ = min(scored, key=lambda s: abs(s[1]))
    if confidence == "낮음":
        return Card(good=f"👍 {good_m.good_text}", fix=None, cue=None,
                    extra=["영상 화질이나 자세 인식이 불안정해서 자세한 지적은 생략했어요"],
                    good_key=good_m.key, confidence=confidence, evidence=evidence)

    fix_m, fix_z, fix_score = max(scored, key=lambda s: s[2])
    if fix_score < 1.5:
        return Card(good=f"👍 {good_m.good_text}", fix=None,
                    cue="👉 평소 좋은 샷과 거의 같은 스윙이에요. 이 느낌 그대로",
                    good_key=good_m.key, confidence=confidence, evidence=evidence)
    if good_m.key == fix_m.key:
        good_m = sorted(scored, key=lambda s: abs(s[1]))[1][0]

    direction_low = fix_z < 0
    fault = fix_m.fault_low if direction_low else fix_m.fault_high
    cue = fix_m.cue_low if direction_low else fix_m.cue_high
    ref_val = reference.get(fix_m.key) if reference is not None else None
    if ref_val is not None and pd.notna(ref_val):
        fault += f" (기준 스윙보다 {_magnitude(fix_m, row[fix_m.key] - ref_val)})"

    extra = []
    flight = row.get("ball_flight_code")
    for link in flight_links.get(fix_m.key, []):
        if link["flight"] == flight and link["sign"] == np.sign(fix_z):
            name = "슬라이스" if flight == 2 else "훅"
            extra.append(f"이 패턴이 클 때 {name}가 더 많았어요 (내 샷 {link['n']}개 분석)")
    if recent_fixes[-2:] == [fix_m.key, fix_m.key]:
        extra.append(f"3번 연속 같은 지적이에요. 드릴: {fix_m.drill}")

    return Card(good=f"👍 {good_m.good_text}{_good_suffix(good_m, row[good_m.key])}",
                fix=f"👉 {fault}", cue=f"다음 샷: \"{cue}\"", extra=extra,
                fix_key=fix_m.key, good_key=good_m.key, confidence=confidence, evidence=evidence)
