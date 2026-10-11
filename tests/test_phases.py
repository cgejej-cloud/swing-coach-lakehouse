import numpy as np
import pandas as pd
import pytest

from swing_coach.phases import PhaseDetectionError, detect

FPS = 60.0


def synthetic_pose(y: np.ndarray, x: np.ndarray | None = None) -> pd.DataFrame:
    """Both wrists (landmarks 15, 16) following the given hand height trace."""
    x = np.full_like(y, 0.5) if x is None else x
    rows = [(f, lm, x[f], y[f], 0.9) for f in range(len(y)) for lm in (15, 16)]
    return pd.DataFrame(rows, columns=["frame", "lm", "x", "y", "vis"])


def swing_trace() -> np.ndarray:
    """Address still until 50, top at 100, impact at 116, finish held from ~135."""
    y = np.full(241, 0.52)
    y[50:101] = np.linspace(0.52, 0.30, 51)          # backswing
    y[100:117] = np.linspace(0.30, 0.52, 17)         # downswing
    y[116:136] = np.linspace(0.52, 0.25, 20)         # follow-through
    y[135:] = 0.25                                    # held finish
    return y


def test_detects_events_in_order_near_truth():
    ph = detect(synthetic_pose(swing_trace()), FPS, 241)
    assert ph.p1 < ph.p4 < ph.p7 < ph.p10
    assert abs(ph.p7 - 116) <= 2
    assert abs(ph.p4 - 100) <= 6
    assert abs(ph.p1 - 50) <= 6
    assert 2.0 < ph.tempo < 5.0


def test_split_backswing_from_occlusion_is_merged():
    """A jitter at the top that dips the hands slightly must not be taken for impact."""
    y = swing_trace()
    y[96:100] = 0.40  # wrists hidden behind the head: estimated lower for a few frames
    ph = detect(synthetic_pose(y), FPS, 241)
    assert abs(ph.p7 - 116) <= 2


def test_no_swing_raises():
    with pytest.raises(PhaseDetectionError):
        detect(synthetic_pose(np.full(241, 0.52)), FPS, 241)
