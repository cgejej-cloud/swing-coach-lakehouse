"""Card image: P1/P4/P7/P10 panels with the golfer's skeleton, the reference
swing as a ghost skeleton, coach lines at the address head/hip position, and the
fix point circled in red."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from swing_coach.metrics import L_HIP, L_SH, NOSE, R_HIP, R_SH, address_pose
from swing_coach.video import frames, probe
from swing_coach.viz import BONES, imwrite

GREEN, GHOST, YELLOW, RED = (60, 220, 90), (255, 255, 255), (255, 210, 0), (255, 60, 60)
PANELS = ["P1", "P4", "P7", "P10"]


def _pts(pose: pd.DataFrame, frame: int, w: int, h: int) -> dict[int, np.ndarray]:
    fr = pose.frame.unique()
    f = fr[np.argmin(np.abs(fr - frame))]
    return {int(r.lm): np.array([r.x * w, r.y * h]) for r in pose[pose.frame == f].itertuples()}


def _anchor(p: dict) -> tuple[np.ndarray, float]:
    """Address hip centre and shoulder width in pixels."""
    return (p[L_HIP] + p[R_HIP]) / 2, abs(p[L_SH][0] - p[R_SH][0])


def _bones(img, p: dict, color, thick: int, alpha: float = 1.0) -> None:
    layer = img.copy()
    for a, b in BONES:
        if a in p and b in p:
            cv2.line(layer, tuple(map(int, p[a])), tuple(map(int, p[b])), color, thick, cv2.LINE_AA)
    if NOSE in p:
        cv2.circle(layer, tuple(map(int, p[NOSE])), thick + 3, color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)


def _dashed_vline(img, x: int, y0: int, y1: int, color) -> None:
    for y in range(y0, y1, 18):
        cv2.line(img, (x, y), (x, min(y + 10, y1)), color, 2, cv2.LINE_AA)


def render_card(video: Path, pose: pd.DataFrame, phases: dict[str, int], out: Path,
                fix_frame: str | None = None, fix_landmark: int | None = None,
                ref_pose: pd.DataFrame | None = None, ref_phases: dict[str, int] | None = None) -> None:
    info = probe(video)
    w, h = info.width, info.height
    to_px = lambda d: {k: v * np.array([w, h]) for k, v in d.items()}  # noqa: E731
    addr = to_px(address_pose(pose, phases["P1"]))
    hip0, sw = _anchor(addr)
    ref_addr = to_px(address_pose(ref_pose, ref_phases["P1"])) if ref_pose is not None else None

    wanted = {phases[k]: k for k in PANELS}
    imgs = {}
    for i, rgb in enumerate(frames(video, info)):
        if i in wanted:
            imgs[wanted[i]] = rgb.copy()
        if len(imgs) == len(PANELS):
            break

    # Crop box around the golfer over all panels.
    allpts = np.vstack([np.array(list(_pts(pose, phases[k], w, h).values())) for k in PANELS])
    x0, y0 = allpts.min(0) - sw * 0.9
    x1, y1 = allpts.max(0) + sw * 0.9
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))

    tiles = []
    for k in PANELS:
        img = imgs.get(k)
        if img is None:
            continue
        p = _pts(pose, phases[k], w, h)
        _dashed_vline(img, int(addr[NOSE][0]), y0, y1, YELLOW)
        _dashed_vline(img, int(hip0[0]), y0, y1, YELLOW)
        if ref_pose is not None:
            rp = _pts(ref_pose, ref_phases[k], w, h)
            rhip, rsw = _anchor(ref_addr)
            aligned = {lm: (v - rhip) / rsw * sw + hip0 for lm, v in rp.items()}
            _bones(img, aligned, GHOST, 4, alpha=0.55)
        _bones(img, p, GREEN, 4)
        if k == fix_frame and fix_landmark in p:
            c = tuple(map(int, p[fix_landmark]))
            cv2.circle(img, c, int(sw * 0.35), RED, 5, cv2.LINE_AA)
            start = (int(addr[fix_landmark][0]), c[1])
            if abs(start[0] - c[0]) > 6:
                cv2.arrowedLine(img, start, c, RED, 5, cv2.LINE_AA, tipLength=0.3)
        tile = img[y0:y1, x0:x1]
        tile = cv2.resize(tile, (360, int(360 * tile.shape[0] / tile.shape[1])))
        border = RED if k == fix_frame else (40, 40, 40)
        tile = cv2.copyMakeBorder(tile, 34, 4, 4, 4, cv2.BORDER_CONSTANT, value=border)
        cv2.putText(tile, k, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
    imwrite(out, np.hstack(tiles))
