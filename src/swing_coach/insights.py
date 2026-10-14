"""Cumulative analysis for the app: gold marts from the lakehouse (Trino) + flight findings.

The app keeps working when the lakehouse is down; this module then reports it as
unavailable instead of failing the feed.
"""

import subprocess
import sys
import threading
from datetime import datetime

from swing_coach import baseline, engine
from swing_coach.coach import BY_KEY

FLIGHT_KO = {1: "스트레이트", 2: "슬라이스", 3: "훅"}
GROUP_KO = {"driver": "드라이버", "iron": "아이언"}


def _trino():
    import trino

    return trino.dbapi.connect(host="localhost", port=8080, user="app", request_timeout=10)


def _rows(cur, sql: str) -> list[dict]:
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def lakehouse() -> dict:
    cur = _trino().cursor()
    trend = _rows(cur, """SELECT CAST(month AS varchar) AS month, club_group, shots, straight_rate,
                                 median_pelvis_sway_top_sw, median_tempo
                          FROM lakehouse.gold.monthly_trend ORDER BY month""")
    fixes = _rows(cur, """SELECT club_group, fix_key, cards, share FROM lakehouse.gold.fix_frequency
                          ORDER BY club_group, cards DESC""")
    rounds = _rows(cur, """SELECT session_label, CAST(first_shot_at AS varchar) AS first_shot_at, club_group, shots,
                                  straight_rate, avg_carry_m
                           FROM lakehouse.gold.round_summary ORDER BY first_shot_at DESC LIMIT 12""")
    runs = _rows(cur, """SELECT run_id, status, seconds, dq_failed, CAST(finished_at AS varchar) AS finished_at
                         FROM lakehouse.ops.pipeline_runs ORDER BY finished_at DESC LIMIT 1""")
    for f in fixes:
        f["name"] = BY_KEY[f["fix_key"]].name if f["fix_key"] in BY_KEY else "지적 없음"
    # Golfzon round records (ADR 007): where strokes go, and what happens from each distance.
    by_par = _rows(cur, """SELECT par, sum(holes) AS holes, sum(avg_strokes * holes) / sum(holes) AS avg_strokes,
                                  sum(strokes_over_par) AS over_par, sum(penalty_strokes) AS penalties,
                                  sum(gir_rate * holes) / sum(holes) AS gir_rate, count(DISTINCT pg_code) AS rounds
                           FROM lakehouse.gold.par_type_summary GROUP BY par ORDER BY par""")
    by_distance = _rows(cur, """SELECT distance_bucket, sum(shots) AS shots,
                                       sum(green_rate * shots) / sum(shots) AS green_rate,
                                       sum(penalty_rate * shots) / sum(shots) AS penalty_rate
                                FROM lakehouse.gold.shot_outcome_by_distance
                                WHERE start_lie_group <> 'tee' GROUP BY distance_bucket ORDER BY distance_bucket""")
    drives = _rows(cur, """SELECT played_at, drives, mean_x_m, sd_x_m, fairway_rate, penalty_rate
                           FROM lakehouse.gold.driver_dispersion ORDER BY played_at DESC""")
    return {"trend": trend, "fixes": fixes, "rounds": rounds, "last_run": runs[0] if runs else None,
            "by_par": by_par, "by_distance": by_distance, "drives": drives}


def findings(store) -> list[dict]:
    """Metrics whose distribution differs between straight and curved shots (p < 0.1)."""
    hist = engine.history(store)
    out = []
    for g in ("driver", "iron"):
        _, table = baseline.flight_links(hist, g)
        for r in table[table.p < 0.1].sort_values("p").itertuples():
            m = BY_KEY[r.metric]
            lower = r.median_curve < r.median_straight
            # Say it in the metric's own words when that direction is a known fault.
            fault = m.fault_low if lower else m.fault_high
            pattern = (fault.replace("어요", "는 경향이 강했어요") if fault
                       else f"'{m.name}' 값이 스트레이트보다 {'작았어요' if lower else '컸어요'}")
            out.append({
                "club_group": g, "metric": m.name, "flight": FLIGHT_KO[r.flight], "p": round(r.p, 3),
                "n_curve": r.n_curve, "n_straight": r.n_straight, "effect": round(r.effect, 2),
                "text": f"{GROUP_KO[g]} {FLIGHT_KO[r.flight]} 샷은 스트레이트 샷보다 — {pattern}",
                "cue": (m.cue_low if lower else m.cue_high) or m.cue_low or m.cue_high,
            })
    return out


class Sync:
    """Runs the batch pipeline in the background; at most one run at a time.

    App tables reach the lake through CDC (pipeline/stream.py), so after uploads only
    pose frames (files) need the batch path: `run.py --only pose_frame`.
    """

    ARGS = ["--only", "pose_frame"]

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.last: dict | None = None

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        started = datetime.now()
        try:
            p = subprocess.run([sys.executable, str(engine.ROOT / "pipeline" / "run.py"), *self.ARGS],
                               capture_output=True, text=True, encoding="utf-8", timeout=900)
            tail = (p.stdout or p.stderr).strip().splitlines()[-1:] or [""]
            self.last = {"started": f"{started:%Y-%m-%d %H:%M:%S}", "exit_code": p.returncode, "summary": tail[0]}
        except Exception as e:  # noqa: BLE001 - report, never crash the app
            self.last = {"started": f"{started:%Y-%m-%d %H:%M:%S}", "exit_code": -1, "summary": str(e)}
        finally:
            self.running = False
