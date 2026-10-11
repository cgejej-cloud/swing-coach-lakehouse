"""Data quality checks between landing and silver, written to lakehouse.ops.dq_results.

Count checks alone miss the "same row count, wrong values" failure (an upstream value
changed without its timestamp), so value checks compare column sums as well.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lake import Trino  # noqa: E402
from transform import SILVER  # noqa: E402

VALUE_COLUMNS = {
    "fact_video": ["carry_m", "ball_flight_code"],
    "fact_swing": ["m_tempo_ratio", "m_head_sway_top_sw", "m_pelvis_sway_top_sw", "m_pelvis_shift_imp_sw"],
    "fact_pose_frame": ["x", "y"],
}


def checks(t: Trino) -> list[tuple[str, str, bool, str]]:
    out = []
    for target, (table, keys, _) in SILVER.items():
        raw_n = t.scalar(f"SELECT count(*) FROM raw.landing.{table}")
        sil_n = t.scalar(f"SELECT count(*) FROM lakehouse.silver.{target}")
        out.append((target, "row_count_matches_landing", raw_n == sil_n, f"landing={raw_n} silver={sil_n}"))

        key_cols = ", ".join(f'"{k}"' for k in keys)
        dups = t.scalar(f"SELECT count(*) FROM (SELECT {key_cols} FROM lakehouse.silver.{target} "
                        f"GROUP BY {key_cols} HAVING count(*) > 1)")
        out.append((target, "unique_key", dups == 0, f"duplicate keys={dups}"))

        for c in VALUE_COLUMNS.get(target, []):
            a = t.scalar(f'SELECT coalesce(sum(CAST("{c}" AS double)), 0) FROM raw.landing.{table}')
            b = t.scalar(f'SELECT coalesce(sum(CAST("{c}" AS double)), 0) FROM lakehouse.silver.{target}')
            out.append((target, f"value_sum_matches:{c}", abs(a - b) <= 1e-6 * max(1.0, abs(a)),
                        f"landing={a:.6f} silver={b:.6f}"))

    bad_order = t.scalar("SELECT count(*) FROM lakehouse.silver.fact_swing WHERE NOT (p1 < p4 AND p4 < p7 AND p7 < p10)")
    out.append(("fact_swing", "phase_order_p1<p4<p7<p10", bad_order == 0, f"violations={bad_order}"))

    # Plausibility: implausible tempos come from phase-detection misses. They are expected in
    # low-confidence swings (the app already withholds advice there) -> warn; in swings the
    # app trusts they would produce wrong advice -> error.
    bad_tempo = t.scalar("SELECT count(*) FROM lakehouse.silver.fact_swing WHERE m_tempo_ratio NOT BETWEEN 1 AND 10")
    out.append(("fact_swing", "warn:tempo_in_1_10", bad_tempo == 0, f"outside={bad_tempo}"))
    bad_trusted = t.scalar("SELECT count(*) FROM lakehouse.silver.fact_swing "
                           "WHERE m_tempo_ratio NOT BETWEEN 1 AND 10 AND m_phase_confidence >= 0.45")
    out.append(("fact_swing", "tempo_in_1_10_when_trusted", bad_trusted == 0, f"outside={bad_trusted}"))

    orphan = t.scalar("SELECT count(*) FROM lakehouse.silver.fact_card c LEFT JOIN lakehouse.silver.fact_video v "
                      "ON v.video_id = c.video_id WHERE v.video_id IS NULL")
    out.append(("fact_card", "card_has_video", orphan == 0, f"orphans={orphan}"))

    neg_carry = t.scalar("SELECT count(*) FROM lakehouse.silver.fact_video WHERE carry_m < 0")
    out.append(("fact_video", "carry_non_negative", neg_carry == 0, f"negative={neg_carry}"))
    return out


def record(t: Trino, run_id: str, results) -> None:
    t.run("CREATE SCHEMA IF NOT EXISTS lakehouse.ops")
    t.run("""CREATE TABLE IF NOT EXISTS lakehouse.ops.dq_results (
               run_id varchar, table_name varchar, check_name varchar, passed boolean,
               detail varchar, checked_at timestamp(6) with time zone)""")
    values = ", ".join(
        f"('{run_id}', '{tb}', '{name}', {str(ok).lower()}, '{detail.replace(chr(39), '')}', current_timestamp)"
        for tb, name, ok, detail in results)
    t.run(f"INSERT INTO lakehouse.ops.dq_results VALUES {values}")
