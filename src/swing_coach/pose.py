"""Per-frame pose extraction for the golfer in a face-on nasmo video.

Screen-golf videos often contain bystanders on the sofa behind the golfer, so
MediaPipe is asked for several poses and the golfer is tracked frame to frame:
start from the person nearest the bottom-centre of the frame (standing on the
mat), then keep the pose whose hip centre is closest to the previous pick.
"""

from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from swing_coach.video import VideoInfo, frames, probe

N_LANDMARKS = 33
L_HIP, R_HIP = 23, 24
# Where the golfer stands in a nasmo frame (normalised x, y of hip centre).
MAT_ANCHOR = np.array([0.5, 0.6])


def _landmarker(model_path: Path, num_poses: int = 3) -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        # Pass bytes: the native loader cannot open non-ASCII Windows paths.
        base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _hip_centre(pose) -> np.ndarray:
    return np.array([(pose[L_HIP].x + pose[R_HIP].x) / 2, (pose[L_HIP].y + pose[R_HIP].y) / 2])


def extract(video_path: Path, model_path: Path) -> tuple[pd.DataFrame, VideoInfo]:
    """Return long-format landmarks: frame, t_ms, lm, x, y, z, vis (x, y normalised 0-1)."""
    info = probe(video_path)
    rows = []
    prev = MAT_ANCHOR
    with _landmarker(model_path) as lm:
        for i, rgb in enumerate(frames(video_path, info)):
            t_ms = int(round(i * 1000 / info.fps))
            result = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
            if not result.pose_landmarks:
                continue
            golfer = min(result.pose_landmarks, key=lambda p: np.linalg.norm(_hip_centre(p) - prev))
            prev = _hip_centre(golfer)
            for k, p in enumerate(golfer):
                rows.append((i, t_ms, k, p.x, p.y, p.z, p.visibility))
    df = pd.DataFrame(rows, columns=["frame", "t_ms", "lm", "x", "y", "z", "vis"])
    return df, info
