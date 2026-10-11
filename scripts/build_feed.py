"""Import the 201 Golfzon nasmo videos into the app DB, analyse them, and write a report.

Usage: uv run --no-sync python scripts/build_feed.py [--reset]
Outputs: data/app.db, data/cards/*.jpg, data/analysis_report.md
"""

import argparse
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_coach import baseline, engine  # noqa: E402
from swing_coach.coach import BY_KEY  # noqa: E402
from swing_coach.phases import PhaseDetectionError  # noqa: E402
from swing_coach.store import Store, default_store  # noqa: E402

FLIGHT_KO = {1: "스트레이트", 2: "슬라이스", 3: "훅"}


def import_golfzon(store: Store) -> None:
    meta = pd.read_csv(ROOT / "nasmo" / "nasmo_all_201.csv", encoding="utf-8-sig")
    meta["shot_at"] = pd.to_datetime(meta.created_at, format="%Y.%m.%d %H:%M:%S")
    for r in meta.sort_values("shot_at").itertuples():
        path = ROOT / "nasmo" / "videos" / f"{r.nasmo_seq}.mp4"
        store.add_video(
            content_hash=hashlib.sha256(path.read_bytes()).hexdigest(), path=str(path), source="golfzon",
            nasmo_seq=int(r.nasmo_seq), session_key=str(r.pg_code),
            session_label=f"{r.shot_at:%m.%d} {r.cc_name}", shot_at=f"{r.shot_at:%Y-%m-%d %H:%M:%S}",
            club=r.club, club_group=baseline.CLUB_GROUP.get(r.club), club_source="golfzon",
            ball_flight_code=int(r.ball_flight_code), carry_m=float(r.carry_m), hole=int(r.hole))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if args.reset:
        (ROOT / "data" / "app.db").unlink(missing_ok=True)
        shutil.rmtree(ROOT / "data" / "cards", ignore_errors=True)

    store = default_store()
    import_golfzon(store)

    with store.conn() as c:
        videos = [dict(r) for r in c.execute("SELECT * FROM videos WHERE source = 'golfzon' ORDER BY shot_at")]
    failed = {}
    for v in videos:
        try:
            engine.compute_swing(store, v)
        except PhaseDetectionError as e:
            failed[v["id"]] = str(e)
            store.finish(v["id"], "failed", str(e))

    ctx = engine.build_context(store)
    for v in videos:
        if v["id"] in failed:
            continue
        engine.make_card_for(store, v, ctx)
        store.finish(v["id"], "done")

    write_report(store, ctx, failed)


def write_report(store: Store, ctx, failed: dict) -> None:
    hist = engine.history(store)
    _, posture_acc, n = baseline.train_club_classifier(hist)
    bundle = ctx.club_model or {}
    lines = [f"# 스윙 분석 리포트", "", f"생성: {datetime.now():%Y-%m-%d %H:%M}", "",
             "## 처리 결과", "",
             f"- 영상 {len(hist) + len(failed)}개 중 구간 감지 성공 {len(hist)}개, 실패 {len(failed)}개",
             f"- 분석 대상(HD·신뢰도 0.5 이상·캐리 이상치 제외): "
             f"{int(baseline.usable(hist[hist.club_group.notna()]).sum())}개", ""]

    lines += ["## 클럽 자동 판별 (드라이버 vs 아이언)", "",
              f"- 기준선: 자세 특징 {len(baseline.CLUB_FEATURES)}개 로지스틱 회귀, 5-fold 정확도 {posture_acc:.1%} (n={n})",
              f"- 사용 모델: `{bundle.get('kind', '-')}` (MobileNetV3 사전학습 임베딩 + 자세 특징), "
              f"5-fold 정확도 **{bundle.get('accuracy_cv', float('nan')):.1%}** (n={bundle.get('n', '-')}) — scripts/train_club.py",
              f"- 확신도 {engine.CLUB_MIN_CONFIDENCE} 미만은 판정 보류: 교차검증에서 샷 91%를 정확도 94.8%로 판별", ""]

    lines += ["## 기준 스윙", ""]
    for g, vid in ctx.refs.items():
        r = hist.set_index("video_id").loc[vid]
        lines.append(f"- {g}: video {vid} (나스모 {int(r.nasmo_seq)}, {r.shot_at}, 캐리 {r.carry_m:.0f}m, 템포 {r.tempo_ratio:.1f}:1)")
    lines.append("")

    lines += ["## 구질과 관련 있는 지표 (스트레이트 vs 휘는 샷, Mann-Whitney)", "",
              "| 클럽 | 지표 | 비교 | n(휜 샷/스트레이트) | 스트레이트 중앙값 | 휜 샷 중앙값 | 효과 크기 | p |",
              "|---|---|---|---|---|---|---|---|"]
    for g in ("driver", "iron"):
        _, table = baseline.flight_links(hist, g)
        if table.empty:
            continue
        for r in table.sort_values("p").itertuples():
            mark = " **" if r.p < 0.1 else ""
            lines.append(f"| {g} | {r.name} | {FLIGHT_KO[r.flight]} | {r.n_curve}/{r.n_straight} | "
                         f"{r.median_straight:.3f} | {r.median_curve:.3f} | {r.effect:+.2f} | {r.p:.3f}{mark} |")
    lines += ["", "p < 0.1 인 지표만 카드의 \"이 패턴이 클 때 훅/슬라이스가 더 많았어요\" 문장에 쓰인다.", ""]

    with store.conn() as c:
        cards = pd.DataFrame([dict(r) for r in c.execute(
            "SELECT c.fix_key, c.confidence, v.club_group FROM cards c JOIN videos v ON v.id = c.video_id")])
    lines += ["## 카드 요약", "", f"- 카드 {len(cards)}장", ""]
    lines += ["| 고칠 점 | 드라이버 | 아이언 |", "|---|---|---|"]
    counts = cards.assign(fix=cards.fix_key.map(lambda k: BY_KEY[k].name if k in BY_KEY else "(지적 없음)")) \
        .pivot_table(index="fix", columns="club_group", aggfunc="size", fill_value=0)
    for name, r in counts.sort_values(counts.columns[0], ascending=False).iterrows():
        lines.append(f"| {name} | {r.get('driver', 0)} | {r.get('iron', 0)} |")
    lines += ["", "신뢰도 분포: " + ", ".join(f"{k} {v}" for k, v in cards.confidence.value_counts().items()), ""]
    if failed:
        lines += ["## 구간 감지 실패", ""] + [f"- video {k}: {v}" for k, v in failed.items()]

    out = ROOT / "data" / "analysis_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
