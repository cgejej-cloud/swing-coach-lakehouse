"""Driver vs iron from the address frame: an image patch around the feet and ball.

Posture alone separates the two clubs only ~80% of the time. At address the ball
and club head sit between the feet: a driver's ball is teed up and its head is
large and round, an iron's ball sits on the mat behind a thin blade. A small
grayscale patch of that region, scaled to the golfer's shoulder width so camera
distance does not matter, carries that difference.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from swing_coach.metrics import L_ANK, L_SH, R_ANK, R_SH, address_pose
from swing_coach.video import frames, probe

PATCH_W, PATCH_H = 32, 32


EMBEDDER = Path(__file__).resolve().parents[2] / "models" / "mobilenet_v3_large_embedder.tflite"
_embedder = None


def embed(rgb_crop: np.ndarray) -> np.ndarray:
    """MobileNetV3 (ImageNet-pretrained, via MediaPipe) embedding of an RGB crop.

    Transfer learning: raw pixels vary with each venue's lighting and mat colour;
    pretrained features describe shapes (teed ball, large round head vs thin blade).
    """
    global _embedder
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if _embedder is None:
        _embedder = vision.ImageEmbedder.create_from_options(vision.ImageEmbedderOptions(
            base_options=BaseOptions(model_asset_buffer=EMBEDDER.read_bytes()), l2_normalize=True))
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_crop))
    return np.asarray(_embedder.embed(img).embeddings[0].embedding, dtype=np.float32)


def address_crop(video: Path, pose: pd.DataFrame, p1: int) -> np.ndarray:
    """RGB crop of the ball / club-head region at address (see address_patch)."""
    return _crop(video, pose, p1, gray=False)


def top_crop(video: Path, pose: pd.DataFrame, p4: int) -> np.ndarray:
    """RGB crop around the golfer at the top of the backswing, where shaft length shows."""
    info = probe(video)
    a = {k: v * np.array([info.width, info.height]) for k, v in address_pose(pose, p4).items()}
    sw = abs(a[L_SH][0] - a[R_SH][0])
    cx = (a[L_SH][0] + a[R_SH][0]) / 2
    cy = (a[L_SH][1] + a[R_SH][1]) / 2
    x0, x1, y0, y1 = int(cx - 3 * sw), int(cx + 3 * sw), int(cy - 3 * sw), int(cy + 1.5 * sw)
    for i, rgb in enumerate(frames(video, info)):
        if i == p4:
            pad = int(4 * sw)
            padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return cv2.resize(padded[y0 + pad: y1 + pad, x0 + pad: x1 + pad], (224, 224), interpolation=cv2.INTER_AREA)
    raise ValueError(f"frame {p4} not found in {video}")


def address_patch(video: Path, pose: pd.DataFrame, p1: int) -> np.ndarray:
    """Grayscale patch (PATCH_H x PATCH_W) between and below the feet at address, standardised."""
    return _crop(video, pose, p1, gray=True)


def _crop(video: Path, pose: pd.DataFrame, p1: int, gray: bool) -> np.ndarray:
    info = probe(video)
    a = {k: v * np.array([info.width, info.height]) for k, v in address_pose(pose, p1).items()}
    sw = abs(a[L_SH][0] - a[R_SH][0])
    cx = (a[L_ANK][0] + a[R_ANK][0]) / 2
    foot_y = max(a[L_ANK][1], a[R_ANK][1])
    # The ball sits in front of the stance line, i.e. well below the ankles in a
    # face-on frame from above: take the region from the ankles down ~3 shoulder widths.
    x0, x1 = int(cx - 1.6 * sw), int(cx + 1.6 * sw)
    y0, y1 = int(foot_y - 0.2 * sw), int(foot_y + 3.0 * sw)
    target = max(0, p1 - 5)  # just before takeaway: club head still behind the ball
    for i, rgb in enumerate(frames(video, info)):
        if i == target:
            pad = int(4 * sw)
            padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            crop = padded[y0 + pad: y1 + pad, x0 + pad: x1 + pad]
            if not gray:
                return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            patch = cv2.resize(gray, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            # Normalise lighting between venues.
            return (patch - patch.mean()) / (patch.std() + 1e-6)
    raise ValueError(f"frame {target} not found in {video}")
