"""Face-on swing metrics measured relative to the address position.

Distances are in shoulder widths at address (SW) so they do not depend on the
camera distance or the golfer's size; this also limits the effect of the
wide-angle lens, which distorts absolute angles more than small relative moves.
Sign convention for x moves: + = toward the target (image +x).
"""

import numpy as np
import pandas as pd

from swing_coach.phases import Phases

FEATURE_VERSION = "v0.1"

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28


def _frame_xy(pose: pd.DataFrame, frame: int) -> dict[int, np.ndarray]:
    """Landmark positions at a frame, falling back to the nearest detected frame."""
    frames = pose.frame.unique()
    f = frames[np.argmin(np.abs(frames - frame))]
    pts = pose[pose.frame == f]
    return {int(r.lm): np.array([r.x, r.y]) for r in pts.itertuples()}


ADDRESS_WINDOW = 20  # frames before P1 (~0.33 s at 60 fps)


def address_pose(pose: pd.DataFrame, p1: int) -> dict[int, np.ndarray]:
    """Median landmark positions over the still address window ending at P1.

    Every distance is divided by the address shoulder width, so a single
    mis-detected frame there would distort the whole swing; the median does not.
    """
    win = pose[(pose.frame >= p1 - ADDRESS_WINDOW) & (pose.frame <= p1)]
    if win.empty:
        return _frame_xy(pose, p1)
    med = win.groupby("lm")[["x", "y"]].median()
    return {int(k): r.to_numpy() for k, r in med.iterrows()}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _mid(p: dict, a: int, b: int) -> np.ndarray:
    return (p[a] + p[b]) / 2


LR_PAIRS = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def fix_sides(pose: pd.DataFrame, address_frame: int) -> pd.DataFrame:
    """Facing the camera, the golfer's left side must be on the image's +x side.

    MediaPipe sometimes labels a face-on golfer as seen from behind, swapping
    every left/right landmark; detect that at address and swap back.
    """
    a = address_pose(pose, address_frame)
    left_x = np.mean([a[l][0] for l, _ in LR_PAIRS[:1] + LR_PAIRS[3:4]])
    right_x = np.mean([a[r][0] for _, r in LR_PAIRS[:1] + LR_PAIRS[3:4]])
    if left_x >= right_x:
        return pose
    swap = {l: r for l, r in LR_PAIRS} | {r: l for l, r in LR_PAIRS}
    return pose.assign(lm=pose.lm.map(lambda k: swap.get(k, k)))


def compute(pose: pd.DataFrame, ph: Phases) -> dict[str, float]:
    pose = fix_sides(pose, ph.p1)
    a = address_pose(pose, ph.p1)
    top = _frame_xy(pose, ph.p4)
    imp = _frame_xy(pose, ph.p7)
    fin = _frame_xy(pose, ph.p10)

    sw = abs(a[L_SH][0] - a[R_SH][0])
    hip_a = _mid(a, L_HIP, R_HIP)
    torso = hip_a[1] - _mid(a, L_SH, R_SH)[1]

    def dx(p: dict, lm_or_pt, ref) -> float:
        pt = p[lm_or_pt] if isinstance(lm_or_pt, int) else lm_or_pt
        return float((pt[0] - ref[0]) / sw)

    return {
        "feature_version": FEATURE_VERSION,
        # tempo
        "backswing_s": round(ph.backswing_s, 3),
        "downswing_s": round(ph.downswing_s, 3),
        "tempo_ratio": round(ph.tempo, 2),
        # setup
        "stance_width_sw": round(float(abs(a[L_ANK][0] - a[R_ANK][0]) / sw), 3),
        # + = trail (right) shoulder lower than lead shoulder, as expected with a driver
        "trail_shoulder_drop_deg": round(float(np.degrees(np.arctan2(a[R_SH][1] - a[L_SH][1], sw))), 1),
        # address posture (also used to tell drivers from irons)
        "hands_addr_x_sw": round(dx(a, _mid(a, L_WR, R_WR), hip_a), 3),
        "hands_addr_drop_sw": round(float((_mid(a, L_WR, R_WR)[1] - hip_a[1]) / sw), 3),
        "torso_len_sw": round(float((hip_a[1] - _mid(a, L_SH, R_SH)[1]) / sw), 3),
        "hands_top_height_sw": round(float((hip_a[1] - _mid(top, L_WR, R_WR)[1]) / sw), 3),
        # backswing / top
        "head_sway_top_sw": round(dx(top, NOSE, a[NOSE]), 3),
        "pelvis_sway_top_sw": round(dx(top, _mid(top, L_HIP, R_HIP), hip_a), 3),
        "lead_arm_top_deg": round(_angle(top[L_SH], top[L_EL], top[L_WR]), 1),
        # impact
        "head_move_imp_sw": round(dx(imp, NOSE, a[NOSE]), 3),
        "pelvis_shift_imp_sw": round(dx(imp, _mid(imp, L_HIP, R_HIP), hip_a), 3),
        "lead_knee_imp_deg": round(_angle(imp[L_HIP], imp[L_KNEE], imp[L_ANK]), 1),
        "hands_ahead_imp_sw": round(dx(imp, _mid(imp, L_WR, R_WR), _mid(a, L_WR, R_WR)), 3),
        # finish
        "pelvis_over_lead_foot_fin_sw": round(dx(fin, _mid(fin, L_HIP, R_HIP), fin[L_ANK]), 3),
        "lead_arm_fin_deg": round(_angle(fin[L_SH], fin[L_EL], fin[L_WR]), 1),
        # quality: shoulder width / torso height in normalised units (x and y are
        # scaled by different frame sides) has median ~0.43 over 186 swings; far
        # outside 0.25-0.7 the body scale, and so every *_sw metric, is unreliable
        "phase_confidence": ph.confidence,
        "scale_ok": int(torso > 0 and 0.25 <= sw / torso <= 0.7),
    }
