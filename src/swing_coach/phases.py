"""Swing phase detection from the hand trajectory of a face-on video.

Image coordinates: y grows downward; a right-handed golfer facing the camera has
the target on the image's +x side (the simulator screen).

Key events (P-system names):
  P1 takeaway start - last still frame before the hands start moving back
  P4 top            - highest hands before impact
  P7 impact         - hands back at the lowest point after the top
  P10 finish        - highest hands after impact
The hands rise twice (backswing, follow-through) with impact in the dip between
them, so events are found from "hands well above address height" segments.
Hand speed is not used: wrists are hidden behind the head at the top and the
resulting landmark jumps produce the largest speeds of the clip.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

L_WRIST, R_WRIST = 15, 16


@dataclass(frozen=True)
class Phases:
    p1: int
    p4: int
    p7: int
    p10: int
    fps: float
    confidence: float

    @property
    def backswing_s(self) -> float:
        return (self.p4 - self.p1) / self.fps

    @property
    def downswing_s(self) -> float:
        return (self.p7 - self.p4) / self.fps

    @property
    def tempo(self) -> float:
        return self.backswing_s / self.downswing_s


def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    win = max(3, win | 1)
    pad = win // 2
    padded = np.pad(a, pad, mode="edge")
    return np.convolve(padded, np.ones(win) / win, mode="valid")


def hand_track(pose: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """Per-frame mean wrist position, interpolated over missed frames."""
    w = pose[pose.lm.isin([L_WRIST, R_WRIST])].groupby("frame")[["x", "y", "vis"]].mean()
    return w.reindex(range(n_frames)).interpolate(limit_direction="both")


class PhaseDetectionError(ValueError):
    pass


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def detect(pose: pd.DataFrame, fps: float, n_frames: int) -> Phases:
    hands = hand_track(pose, n_frames)
    win = int(round(fps / 12))  # ~80 ms
    x = _smooth(hands.x.to_numpy(), win)
    y = _smooth(hands.y.to_numpy(), win)

    # Address height: the clip opens with the golfer standing still over the ball.
    address_y = float(np.median(y[: int(0.6 * fps)]))
    min_len = int(0.05 * fps)
    # "Well above address": scaled to the swing's height range because the golfer's
    # size in the frame differs between venues (camera distance).
    amplitude = address_y - float(y.min())
    threshold = address_y - max(0.08, 0.45 * amplitude)
    high = [r for r in _runs(y < threshold) if r[1] - r[0] >= min_len]
    # Occlusion at the top can split the backswing into pieces; a real impact dip
    # brings the hands most of the way back to address height, so merge shallower dips.
    merged = high[:1]
    for seg in high[1:]:
        dip = float(y[merged[-1][1]: seg[0] + 1].max())
        if dip < address_y - 0.3 * amplitude:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    if len(merged) < 2:
        raise PhaseDetectionError(f"expected backswing and follow-through, found {len(merged)} high segment(s)")
    back, through = merged[0], merged[1]

    p7 = back[1] + int(np.argmax(y[back[1]: through[0] + 1]))
    p10 = _finish(y, through, fps, amplitude)
    p4 = _top(y, back, p7, fps, amplitude)

    # Takeaway: walk back from the top while the hands are still moving.
    still = np.hypot(np.gradient(x), np.gradient(y)) * fps < 0.1
    p1 = back[0]
    while p1 > 0 and not (still[p1] and abs(y[p1] - address_y) < 0.015):
        p1 -= 1

    confidence = _confidence(hands.vis.to_numpy(), p1, p4, p7, fps)
    return Phases(int(p1), int(p4), int(p7), int(p10), fps, confidence)


# Tuned against 30 hand-labelled clips (scripts/tune_phases.py, label/phase_labels.csv):
# top MAE 5.7 -> 1.7 frames at 60 fps. The finish barely moved with FINISH_BAND (MAE ~9
# frames); the held finish pose changes little over those frames, so it is left as is.
TOP_SMOOTH_S = 0.35
TOP_BAND = 0.06     # fraction of swing height: latest frame this close to the highest point
FINISH_BAND = 0.02


def _top(y: np.ndarray, back: tuple[int, int], p7: int, fps: float, amplitude: float) -> int:
    """Top of the backswing: the latest frame before impact with the hands (smoothed)
    within TOP_BAND of their highest point.

    The wrists are hidden behind the head at the top, so the raw minimum jitters and
    tends to land before the club actually changes direction.
    """
    ys = _smooth(y, int(TOP_SMOOTH_S * fps))
    seg = ys[back[0]: p7]
    near = np.flatnonzero(seg <= seg.min() + TOP_BAND * amplitude)
    return back[0] + int(near.max())


def _finish(y: np.ndarray, through: tuple[int, int], fps: float, amplitude: float) -> int:
    """Finish: the latest frame of the follow-through with the hands within FINISH_BAND
    of their highest point, i.e. where the held finish ends rather than where it starts."""
    seg = y[through[0]: through[1] + 1]
    near = np.flatnonzero(seg <= seg.min() + FINISH_BAND * amplitude)
    return through[0] + int(near.max())


def _confidence(vis: np.ndarray, p1: int, p4: int, p7: int, fps: float) -> float:
    """0-1 score: wrist visibility around the events times timing plausibility."""
    window = np.concatenate([vis[max(0, f - 3): f + 4] for f in (p1, p4, p7)])
    score = float(np.nanmean(window))
    back, down = (p4 - p1) / fps, (p7 - p4) / fps
    if not 0.4 <= back <= 2.0:
        score *= 0.5
    if not 0.12 <= down <= 0.6:
        score *= 0.5
    return round(score, 3)
