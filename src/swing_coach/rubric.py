"""Judge a swing against textbook ("정석") face-on fundamentals: 잘함 / 보통 / 아쉬움.

coach.make_card compares a swing with the golfer's OWN good shots, so a habit that is
always there looks normal. This module compares it with what good players do instead,
using ranges collected from biomechanics studies and motion-capture data (docs/adr/006).

Sign convention matches metrics.py: x is + toward the target (right-handed golfer seen
face-on). Sources mostly report sway as + away from the target; those were flipped.
Bands are in shoulder widths (SW, shoulder joint centres at address, ~40 cm).

Only CORE metrics (strongest evidence, least distorted face-on) decide the verdict.
SOFT metrics are shown as guidance: sources disagree, or the 2D measurement is biased
(e.g. the head is tracked by the nose, which also moves when the head turns).
"""

from dataclasses import dataclass

import numpy as np

from swing_coach.coach import BALL_CM, BY_KEY, SHOULDER_CM


@dataclass(frozen=True)
class Band:
    good: tuple[float, float]
    ok: tuple[float, float]      # amber outside good but inside ok; red outside ok


@dataclass(frozen=True)
class Rule:
    key: str
    core: dict                   # club_group -> bool (core decides the verdict)
    bands: dict                  # club_group -> Band
    fault_low: str
    fault_high: str
    cue_low: str
    cue_high: str
    source: str


def _same(band: Band) -> dict:
    return {"driver": band, "iron": band}


RULES = [
    Rule("tempo_ratio", {"driver": True, "iron": True}, _same(Band((2.7, 3.3), (2.4, 3.8))),
         "백스윙에 비해 다운스윙이 길어요, 서두르는 리듬이에요",
         "백스윙이 느리거나 탑에서 멈칫해요",
         "백스윙은 천천히, 탑에서 한 박자", "탑에서 끊지 말고 이어서 내려오기",
         "Novosel Tour Tempo (투어 3:1, 21/7·24/8·27/9 프레임); Grober 센서 연구"),
    Rule("pelvis_sway_top_sw", {"driver": True, "iron": True},
         {"driver": Band((-0.08, 0.05), (-0.15, 0.10)), "iron": Band((-0.06, 0.05), (-0.12, 0.10))},
         "탑에서 골반이 오른쪽으로 밀렸어요", "탑에서 골반이 목표 쪽으로 먼저 넘어갔어요",
         "오른 무릎을 버티고 골반은 제자리에서 돌리기", "백스윙 때 체중은 오른발 안쪽에",
         "GOLFTEC 투어 데이터 (탑 골반 좌우 이동 약 0.5cm); TPI 스웨이 정의"),
    Rule("pelvis_shift_imp_sw", {"driver": True, "iron": True},
         {"driver": Band((0.10, 0.35), (0.05, 0.45)), "iron": Band((0.15, 0.40), (0.05, 0.50))},
         "임팩트 때 골반이 왼쪽으로 덜 넘어왔어요", "임팩트 때 골반이 목표 쪽으로 너무 밀려 나갔어요",
         "다운스윙은 왼쪽 골반부터 시작", "골반은 밀지 말고 왼쪽으로 돌리기",
         "GOLFTEC (투어 임팩트 골반 약 4cm 목표 쪽); Sportsbox 투어 선수 예시"),
    Rule("pelvis_over_lead_foot_fin_sw", {"driver": True, "iron": True},
         _same(Band((-0.15, 0.15), (-0.25, 0.25))),
         "피니시에서 체중이 오른쪽에 남았어요", "피니시에서 몸이 왼발 바깥으로 넘어갔어요",
         "피니시는 왼발 위에 서기", "피니시에서 3초 균형 잡고 서기",
         "BodiTrak·GolfWRX 압력판 (투어 임팩트 왼발 75~90%)"),
    Rule("hands_ahead_imp_sw", {"driver": False, "iron": True},
         {"driver": Band((0.0, 0.15), (-0.05, 0.25)), "iron": Band((0.10, 0.30), (0.0, 0.45))},
         "임팩트 때 손이 공보다 뒤에 있어요", "임팩트 때 손이 너무 앞서 나갔어요",
         "손이 공보다 먼저 지나가게", "손과 클럽헤드가 함께 지나가게",
         "TrackMan 투어 평균 (6번 아이언 로프트 약 10° 세워 맞음)"),
    Rule("stance_width_sw", {"driver": False, "iron": False},
         {"driver": Band((1.15, 1.35), (0.9, 1.5)), "iron": Band((0.95, 1.2), (0.85, 1.5))},
         "스탠스가 정석보다 좁아요", "스탠스가 정석보다 넓어요",
         "발을 조금 더 벌려 서기", "발 간격을 조금 좁히기",
         "ISBS 연구 (드라이버 키의 30%, 5번 아이언 27%); TPI는 어깨너비 이내 권장"),
    Rule("trail_shoulder_drop_deg", {"driver": False, "iron": False},
         {"driver": Band((6, 14), (0, 18)), "iron": Band((3, 10), (0, 14))},
         "어드레스에서 어깨가 너무 평평해요", "어드레스에서 오른쪽 어깨가 많이 내려갔어요",
         "오른쪽 어깨를 살짝 낮춰 셋업", "어깨 높이를 조금 더 평평하게",
         "GOLFTEC (상위 선수 셋업 어깨 기울기 약 8°)"),
    Rule("head_sway_top_sw", {"driver": False, "iron": False},
         {"driver": Band((-0.20, 0.0), (-0.30, 0.05)), "iron": Band((-0.12, 0.0), (-0.25, 0.05))},
         "탑에서 머리가 오른쪽으로 밀렸어요", "탑에서 머리가 목표 쪽으로 기울었어요",
         "머리는 공 뒤에 두고 어깨만 돌리기", "백스윙 때 머리는 제자리",
         "Byrne & Kenny 2008 (상급자 머리 이동 약 2cm vs 하급자 12cm)"),
    Rule("head_move_imp_sw", {"driver": False, "iron": False},
         {"driver": Band((-0.20, 0.0), (-0.30, 0.10)), "iron": Band((-0.12, 0.08), (-0.20, 0.15))},
         "임팩트 때 머리가 너무 뒤에 남았어요", "임팩트 때 머리가 공 쪽으로 따라 나갔어요",
         "임팩트 때 체중을 왼쪽으로", "임팩트까지 머리는 공 뒤에",
         "Sportsbox 투어 예시 (임팩트 때 상체가 골반보다 뒤)"),
    Rule("lead_knee_imp_deg", {"driver": False, "iron": False},
         {"driver": Band((160, 181), (150, 181)), "iron": Band((155, 181), (150, 181))},
         "임팩트 때 왼무릎이 굽어 있어요", "",
         "임팩트 때 왼다리로 버티기", "",
         "UConn 연구 (임팩트 왼무릎 굽힘 약 10°)"),
]
BY_RULE = {r.key: r for r in RULES}
LEVEL = {0: "green", 1: "amber", 2: "red"}


@dataclass
class Check:
    key: str
    value: float
    status: str          # green / amber / red
    direction: str       # low / high / ""
    severity: float      # 0 inside good, 0-1 amber, >1 red
    gap: float           # distance from the nearest edge of the good band
    core: bool


def check(rule: Rule, group: str, value: float) -> Check:
    band = rule.bands[group]
    lo, hi = band.good
    if lo <= value <= hi:
        return Check(rule.key, value, "green", "", 0.0, 0.0, rule.core[group])
    direction = "low" if value < lo else "high"
    edge, outer = (lo, band.ok[0]) if direction == "low" else (hi, band.ok[1])
    gap = abs(value - edge)
    severity = gap / max(abs(edge - outer), 1e-6)
    return Check(rule.key, value, "amber" if severity <= 1 else "red", direction,
                 round(severity, 2), gap, rule.core[group])


@dataclass
class Verdict:
    level: str           # 잘함 / 보통 / 아쉬움 / 판정 보류
    checks: list[Check]

    @property
    def core(self) -> list[Check]:
        return [c for c in self.checks if c.core]


def judge(row, group: str | None, confidence: str = "높음") -> Verdict:
    """row: mapping with the metric values of one swing."""
    if group not in ("driver", "iron"):
        return Verdict("판정 보류", [])
    checks = []
    for rule in RULES:
        v = row.get(rule.key) if hasattr(row, "get") else None
        if v is None or not np.isfinite(v):
            continue
        checks.append(check(rule, group, float(v)))
    core = [c for c in checks if c.core]
    if confidence == "낮음" or len(core) < 3:
        return Verdict("판정 보류", checks)
    red = sum(c.status == "red" for c in core)
    amber = sum(c.status == "amber" for c in core)
    level = "아쉬움" if red >= 2 else ("잘함" if red == 0 and amber <= 1 else "보통")
    return Verdict(level, checks)


def _magnitude(key: str, gap: float) -> str:
    unit = BY_KEY[key].unit
    if unit == "sw":
        return f"공 {max(1, round(gap * SHOULDER_CM / BALL_CM))}개 정도"
    if unit == "deg":
        return f"{gap:.0f}°"
    return f"{gap:.1f}"


def textbook_points(verdict: Verdict, group: str) -> tuple[Check | None, Check | None]:
    """(fix, good): the worst core check outside the good band (soft only when red), and
    the best core check inside it."""
    order = [r.key for r in RULES]
    bad = [c for c in verdict.checks if c.status != "green" and (c.core or c.status == "red")]
    fix = max(bad, key=lambda c: (c.core, c.severity), default=None)
    greens = [c for c in verdict.core if c.status == "green"]
    good = min(greens, key=lambda c: order.index(c.key), default=None)
    return fix, good


def fault_text(c: Check, group: str) -> tuple[str, str]:
    """(fault, cue) for a check outside the good band, quoting the textbook range."""
    rule = BY_RULE[c.key]
    lo, hi = rule.bands[group].good
    fault = rule.fault_low if c.direction == "low" else rule.fault_high
    cue = rule.cue_low if c.direction == "low" else rule.cue_high
    if BY_KEY[c.key].unit == "ratio":
        fault += f" ({c.value:.1f}:1, 정석 {lo:.1f}~{hi:.1f}:1)"
    else:
        fault += f" (정석 범위보다 {_magnitude(c.key, c.gap)})"
    return fault, cue


def good_text(c: Check) -> str:
    m = BY_KEY[c.key]
    if m.unit == "ratio":
        return f"{m.good_text} ({c.value:.1f}:1, 정석 범위)"
    return f"{m.good_text} (정석 범위)"


def apply_to_card(card, row, group: str | None, flight_links: dict, recent_fixes: list[str]) -> Verdict:
    """Judge the swing and rewrite the card's good point / fix / cue against the textbook.

    The personal card (coach.make_card) stays as is when the verdict is withheld
    (low confidence, unknown club), since its comparison with the golfer's own shots
    is still informative there.
    """
    verdict = judge(row, group, card.confidence)
    card.evidence["_verdict"] = as_evidence(verdict, group)
    if verdict.level == "판정 보류":
        return verdict
    fix, good = textbook_points(verdict, group)
    if good:
        card.good, card.good_key = f"👍 {good_text(good)}", good.key
    card.extra = []
    if fix is None:
        card.fix, card.fix_key = None, None
        card.cue = "👉 정석 범위 안의 스윙이에요. 이 느낌 그대로"
        return verdict
    fault, cue = fault_text(fix, group)
    card.fix, card.cue, card.fix_key = f"👉 {fault}", f"다음 샷: \"{cue}\"", fix.key
    sign = -1 if fix.direction == "low" else 1
    flight = row.get("ball_flight_code")
    for link in flight_links.get(fix.key, []):
        if link["flight"] == flight and link["sign"] == sign:
            name = "슬라이스" if flight == 2 else "훅"
            card.extra.append(f"이 패턴이 클 때 {name}가 더 많았어요 (내 샷 {link['n']}개 분석)")
    if recent_fixes[-2:] == [fix.key, fix.key]:
        card.extra.append(f"3번 연속 같은 지적이에요. 드릴: {BY_KEY[fix.key].drill}")
    return verdict


def as_evidence(verdict: Verdict, group: str | None) -> dict:
    return {"level": verdict.level, "group": group,
            "checks": {c.key: {"value": round(c.value, 3), "status": c.status, "core": c.core,
                               "good": list(BY_RULE[c.key].bands[group].good)}
                       for c in verdict.checks}}
