"""Baselines built from the golfer's own history: good shots, reference swing,
metric <-> ball-flight links, and the driver/iron classifier."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from swing_coach.coach import CATALOG

CLUB_GROUP = {"DRIVER": "driver", "IRON": "iron"}
STRAIGHT, SLICE, HOOK = 1, 2, 3
CLUB_FEATURES = ["stance_width_sw", "trail_shoulder_drop_deg", "hands_addr_x_sw", "hands_addr_drop_sw",
                 "torso_len_sw", "hands_top_height_sw", "backswing_s", "downswing_s"]


def usable(df: pd.DataFrame) -> pd.Series:
    """HD video, phases found with reasonable confidence, carry not an outlier for its club."""
    ok = (df.width >= 1280) & df.p4.notna() & (df.phase_confidence >= 0.5)
    if "scale_ok" in df:
        ok &= df.scale_ok.fillna(0) == 1
    q1 = df.groupby("club_group").carry_m.transform(lambda s: s.quantile(0.25))
    q3 = df.groupby("club_group").carry_m.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    return ok & df.carry_m.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def good_shots(df: pd.DataFrame, group: str, exclude: int | None = None) -> pd.DataFrame:
    g = df[(df.club_group == group) & usable(df) & (df.ball_flight_code == STRAIGHT)]
    return g[g.video_id != exclude] if exclude is not None else g


def reference(df: pd.DataFrame, group: str) -> pd.Series:
    """Initial reference: the longest straight shot among confident good shots."""
    g = good_shots(df, group)
    g = g[g.phase_confidence >= 0.6] if (g.phase_confidence >= 0.6).any() else g
    return g.loc[g.carry_m.idxmax()]


def flight_links(df: pd.DataFrame, group: str, alpha: float = 0.1, min_n: int = 5) -> tuple[dict, pd.DataFrame]:
    """Metrics whose distribution differs between straight and slice/hook shots."""
    g = df[(df.club_group == group) & usable(df)]
    straight = g[g.ball_flight_code == STRAIGHT]
    links, rows = {}, []
    for m in CATALOG:
        for flight in (SLICE, HOOK):
            curve = g[g.ball_flight_code == flight][m.key].dropna()
            s = straight[m.key].dropna()
            if len(curve) < min_n or len(s) < min_n:
                continue
            p = float(mannwhitneyu(curve, s).pvalue)
            diff = float(curve.median() - s.median())
            # Effect size: difference in medians over the straight shots' robust spread.
            spread = float((s - s.median()).abs().median() * 1.4826) or np.nan
            rows.append({"group": group, "metric": m.key, "name": m.name, "flight": flight,
                         "n_curve": len(curve), "n_straight": len(s), "median_straight": s.median(),
                         "median_curve": curve.median(), "effect": diff / spread, "p": p})
            if p < alpha:
                links.setdefault(m.key, []).append(
                    {"flight": flight, "sign": float(np.sign(diff)), "n": len(curve) + len(s), "p": round(p, 3)})
    return links, pd.DataFrame(rows)


def train_club_classifier(df: pd.DataFrame):
    """Driver vs iron from address/backswing posture, trained on Golfzon club labels only."""
    data = df[(df.club_source == "golfzon") & df.club_group.notna() & df.p4.notna()].dropna(subset=CLUB_FEATURES)
    X, y = data[CLUB_FEATURES].to_numpy(), (data.club_group == "driver").to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    acc = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    model.fit(X, y)
    return model, float(acc.mean()), len(data)


def predict_club(model, row: pd.Series) -> tuple[str, float]:
    p = float(model.predict_proba(row[CLUB_FEATURES].to_numpy(dtype=float).reshape(1, -1))[0, 1])
    return ("driver", p) if p >= 0.5 else ("iron", 1 - p)
