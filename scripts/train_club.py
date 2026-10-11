"""Train and compare driver/iron classifiers on Golfzon club labels.

Candidates: posture features, raw address patch, pretrained MobileNetV3 embedding of the
ball/club-head region, and combinations. 5-fold stratified CV accuracy decides.
Usage: uv run --no-sync python scripts/train_club.py [--save]
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_coach import engine  # noqa: E402
from swing_coach.baseline import CLUB_FEATURES  # noqa: E402
from swing_coach.club import address_crop, address_embedding, embed, top_crop  # noqa: E402
from swing_coach.store import Store, default_store  # noqa: E402

CACHE = ROOT / "data" / "club_embeddings.npz"


def features(v: dict, p1: int, p4: int) -> np.ndarray:
    """Address ball-region embedding + top-of-backswing embedding (shared with the app)."""
    pose = engine.load_pose(v)
    return np.concatenate([embed(address_crop(Path(v["path"]), pose, p1)),
                           embed(top_crop(Path(v["path"]), pose, p4))])


def load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rows, [single-frame address emb | top emb], and the 3-frame mean address emb."""
    store = default_store()
    hist = engine.history(store)
    hist = hist[(hist.club_source == "golfzon") & hist.club_group.isin(["driver", "iron"])].dropna(subset=CLUB_FEATURES)
    cached = dict(np.load(CACHE)) if CACHE.exists() else {}
    with store.conn() as c:
        videos = {r["id"]: dict(r) for r in c.execute("SELECT * FROM videos")}
    emb, avg = [], []
    for r in hist.itertuples():
        v = videos[r.video_id]
        key = "v2-" + v["content_hash"][:16]
        if key not in cached:
            cached[key] = features(v, int(r.p1), int(r.p4))
        akey = "avg3-" + v["content_hash"][:16]
        if akey not in cached:
            cached[akey] = address_embedding(Path(v["path"]), engine.load_pose(v), int(r.p1))
        emb.append(cached[key])
        avg.append(cached[akey])
    np.savez_compressed(CACHE, **cached)
    return hist.reset_index(drop=True), np.vstack(emb), np.vstack(avg)


def candidates(posture: np.ndarray, emb: np.ndarray, avg: np.ndarray) -> dict:
    lr = lambda c=1.0: LogisticRegression(C=c, max_iter=5000)  # noqa: E731
    half = emb.shape[1] // 2
    out = {
        "posture": (make_pipeline(StandardScaler(), lr()), posture),
        "address_emb+posture": (make_pipeline(StandardScaler(), PCA(40, random_state=0), lr(0.3)),
                                np.hstack([emb[:, :half], posture])),
        "address+top_emb+posture": (make_pipeline(StandardScaler(), PCA(48, random_state=0), lr(0.3)),
                                    np.hstack([emb, posture])),
    }
    for pcs, c in ((24, 0.3), (40, 0.3), (40, 0.1), (64, 0.1)):
        out[f"address_avg3_emb+posture[pca{pcs},C{c}]"] = (
            make_pipeline(StandardScaler(), PCA(pcs, random_state=0), lr(c)), np.hstack([avg, posture]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    hist, emb, avg = load()
    y = (hist.club_group == "driver").to_numpy()
    posture = hist[CLUB_FEATURES].to_numpy(dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    scores = {}
    for name, (model, X) in candidates(posture, emb, avg).items():
        pred = cross_val_predict(model, X, y, cv=cv)
        scores[name] = (pred == y).mean()
        wrong = hist.loc[pred != y]
        print(f"{name:38s} acc={scores[name]:.1%}  errors={len(wrong)} (low-res: {(wrong.width < 640).sum()})")

    best = max(scores, key=scores.get)
    print(f"best: {best} ({scores[best]:.1%}, n={len(y)})")

    # Abstain when unsure: accuracy on the shots we do label vs how many we label.
    model, X = candidates(posture, emb, avg)[best]
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    conf = np.maximum(proba, 1 - proba)
    for th in (0.5, 0.6, 0.7, 0.8):
        keep = conf >= th
        acc = ((proba[keep] >= 0.5) == y[keep]).mean()
        print(f"  confidence >= {th}: labelled {keep.mean():.0%} of shots, accuracy {acc:.1%}")
    if args.save:
        model, X = candidates(posture, emb, avg)[best]
        model.fit(X, y)
        joblib.dump({"kind": best, "model": model, "accuracy_cv": float(scores[best]), "n": int(len(y)),
                     "posture_features": CLUB_FEATURES,
                     # which address embedding the app must compute for this model
                     "address_features": "avg3" if "avg3" in best else "single"},
                    ROOT / "models" / "club_clf.joblib")
        print("saved models/club_clf.joblib")


if __name__ == "__main__":
    main()
