"""Debug rendering: skeleton overlays and contact sheets."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# MediaPipe pose connections used for the body (face points omitted).
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]


def draw_skeleton(img: np.ndarray, pts: pd.DataFrame, color=(0, 255, 0)) -> np.ndarray:
    """pts: rows for one frame with columns lm, x, y (normalised)."""
    h, w = img.shape[:2]
    xy = {int(r.lm): (int(r.x * w), int(r.y * h)) for r in pts.itertuples()}
    for a, b in BONES:
        if a in xy and b in xy:
            cv2.line(img, xy[a], xy[b], color, 3, cv2.LINE_AA)
    if 0 in xy:
        cv2.circle(img, xy[0], 6, (0, 0, 255), -1)
    return img


def contact_sheet(images: list[np.ndarray], labels: list[str], cols: int = 4, width: int = 360) -> np.ndarray:
    tiles = []
    for img, label in zip(images, labels):
        h, w = img.shape[:2]
        t = cv2.resize(img, (width, int(h * width / w)))
        cv2.putText(t, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def imwrite(path: Path, img_rgb: np.ndarray) -> None:
    """cv2.imwrite cannot handle non-ASCII paths on Windows, so encode in memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    path.write_bytes(buf.tobytes())
