"""Video -> swing metrics -> teaching card. Shared by the batch import and the app worker."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from swing_coach import baseline, llm, rubric
from swing_coach.club import address_crop, address_embedding, embed
from swing_coach.coach import BY_KEY, Card, make_card
from swing_coach.metrics import compute, fix_sides
from swing_coach.phases import NotFullSwing, detect
from swing_coach.pose import extract
from swing_coach.render import render_card
from swing_coach.store import Store

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "pose_landmarker_full.task"
POSE_DIR = ROOT / "data" / "pose"
CARD_DIR = ROOT / "data" / "cards"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pose_path(v) -> Path:
    # Keyed by content, not by row id: ids are reused after the DB is rebuilt.
    return POSE_DIR / (f"{v['nasmo_seq']}.parquet" if v["nasmo_seq"] else f"h{v['content_hash'][:16]}.parquet")


def load_pose(v) -> pd.DataFrame:
    path = pose_path(v)
    if path.exists():
        return pd.read_parquet(path)
    df, _ = extract(Path(v["path"]), MODEL)
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def compute_swing(store: Store, v) -> None:
    from swing_coach.video import probe

    info = probe(Path(v["path"]))
    pose = load_pose(v)
    ph = detect(pose, info.fps, info.n_frames)
    metrics = compute(pose, ph)
    store.update_video(v["id"], width=info.width, fps=info.fps)
    store.save_swing(v["id"], {"P1": ph.p1, "P4": ph.p4, "P7": ph.p7, "P10": ph.p10}, metrics)


def history(store: Store) -> pd.DataFrame:
    with store.conn() as c:
        rows = c.execute("""
            SELECT v.id AS video_id, v.nasmo_seq, v.club, v.club_group, v.club_source, v.ball_flight_code,
                   v.carry_m, v.width, v.shot_at, v.session_key, v.seq_no, s.p1, s.p4, s.p7, s.p10, s.metrics_json
            FROM videos v JOIN swings s ON s.video_id = v.id""").fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    metrics = pd.DataFrame([json.loads(m) for m in df.pop("metrics_json")])
    return pd.concat([df, metrics], axis=1)


@dataclass
class Context:
    hist: pd.DataFrame
    links: dict[str, dict]
    refs: dict[str, int]
    club_model: object | None


def build_context(store: Store) -> Context:
    hist = history(store)
    links = {g: baseline.flight_links(hist, g)[0] for g in ("driver", "iron")}
    refs = store.references()
    for g in ("driver", "iron"):
        if g not in refs and not baseline.good_shots(hist, g).empty:
            ref = baseline.reference(hist, g)
            store.set_reference(g, int(ref.video_id))
            refs[g] = int(ref.video_id)
    model = joblib.load(CLUB_MODEL) if CLUB_MODEL.exists() else None
    return Context(hist, links, refs, model)


CLUB_MODEL = ROOT / "models" / "club_clf.joblib"
# Below this the club is left unknown: on cross-validation (3-frame address embedding +
# posture), 93% of shots clear it with 95.5% accuracy, versus 93.5% when every shot is
# forced to a label. Picked among 7 variants on the same folds, so slightly optimistic.
CLUB_MIN_CONFIDENCE = 0.8


def assign_club(store: Store, v, ctx: Context) -> str | None:
    """Golfzon label when present, otherwise predict driver/iron (scripts/train_club.py model)."""
    if v["club_group"]:
        return v["club_group"]
    if v["club"]:
        return baseline.CLUB_GROUP.get(v["club"])
    if ctx.club_model is None:
        return None
    row = ctx.hist.set_index("video_id").loc[v["id"]]
    pose = load_pose(v)
    if ctx.club_model.get("address_features") == "avg3":
        emb = address_embedding(Path(v["path"]), pose, int(row["p1"]))
    else:
        emb = embed(address_crop(Path(v["path"]), pose, int(row["p1"])))
    x = np.concatenate([emb, row[ctx.club_model["posture_features"]].to_numpy(dtype=float)]).reshape(1, -1)
    p_driver = float(ctx.club_model["model"].predict_proba(x)[0, 1])
    group, conf = ("driver", p_driver) if p_driver >= 0.5 else ("iron", 1 - p_driver)
    if conf < CLUB_MIN_CONFIDENCE:
        store.update_video(v["id"], club_source="uncertain", club_conf=round(conf, 3))
        return None
    store.update_video(v["id"], club_group=group, club_source="predicted", club_conf=round(conf, 3))
    ctx.hist.loc[ctx.hist.video_id == v["id"], "club_group"] = group
    return group


def make_card_for(store: Store, v, ctx: Context) -> None:
    hist = ctx.hist
    row = hist.set_index("video_id").loc[v["id"]].copy()
    row["width"] = row.get("width") or 1280
    group = assign_club(store, v, ctx)
    phases = {k.upper(): int(row[k]) for k in ("p1", "p4", "p7", "p10")}
    pose = fix_sides(load_pose(v), phases["P1"])
    out = CARD_DIR / f"{v['id']}.jpg"

    if group is None:
        uncertain = v["source"] == "upload"
        card = Card(good=("클럽(드라이버/아이언)을 확실히 알아보지 못해서 자세 표시만 했어요" if uncertain
                          else "이번 샷은 드라이버·아이언이 아니라서 자세 표시만 했어요"),
                    fix=None, cue=None, confidence="-")
        render_card(Path(v["path"]), pose, phases, out)
        store.save_card(v["id"], card, _rel(out))
        return

    good = baseline.good_shots(hist, group, exclude=v["id"])
    ref_id = ctx.refs.get(group)
    reference = hist.set_index("video_id").loc[ref_id] if ref_id and ref_id != v["id"] else None
    # "N in a row" only counts within the same round / upload batch.
    prior = hist[(hist.club_group == group) & (hist.session_key == row["session_key"])
                 & (hist.shot_at < row["shot_at"])].sort_values("shot_at")
    with store.conn() as c:
        fixes = {r["video_id"]: r["fix_key"] for r in c.execute("SELECT video_id, fix_key FROM cards")}
    recent = [fixes.get(i) for i in prior.video_id.tail(2)]

    card = make_card(row, good, reference, ctx.links.get(group, {}), recent)
    # Judge against textbook fundamentals; this replaces the fix chosen against own shots.
    rubric.apply_to_card(card, row, group, ctx.links.get(group, {}), recent)
    if card.fix and llm.available():
        # Rules choose the content; the LLM may only reword it (guarded in llm.polish_card).
        _, card.evidence["_llm"] = llm.polish_card(card)

    ref_pose = ref_phases = None
    if reference is not None:
        with store.conn() as c:
            rv = dict(c.execute("SELECT * FROM videos WHERE id = ?", (ref_id,)).fetchone())
        ref_phases = {k.upper(): int(reference[k]) for k in ("p1", "p4", "p7", "p10")}
        ref_pose = fix_sides(load_pose(rv), ref_phases["P1"])
    m = BY_KEY.get(card.fix_key)
    render_card(Path(v["path"]), pose, phases, out,
                fix_frame=m.frame if m else None, fix_landmark=m.landmark if m else None,
                ref_pose=ref_pose, ref_phases=ref_phases)
    store.save_card(v["id"], card, _rel(out))


def note_not_full_swing(store: Store, v) -> None:
    """Putts and chips are valid uploads, not failures: say so on the card."""
    card = Card(good="풀스윙이 아닌 샷(퍼팅·어프로치)이라 자세 분석은 하지 않았어요", fix=None, cue=None,
                confidence="-")
    store.save_card(v["id"], card, None)
    store.finish(v["id"], "done")


def process(store: Store, v) -> None:
    """Full pipeline for one queued video (used by the upload worker)."""
    try:
        compute_swing(store, v)
    except NotFullSwing:
        note_not_full_swing(store, v)
        return
    ctx = build_context(store)
    make_card_for(store, v, ctx)
