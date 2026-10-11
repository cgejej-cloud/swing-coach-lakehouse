"""SQL guard and answer-grounding checks of the text-to-SQL agent (no model, no database)."""

from swing_coach.ask import _numbers_ok, validate_sql


def test_plain_select_on_allowed_tables_passes():
    assert validate_sql("SELECT count(*) FROM lakehouse.silver.fact_video WHERE club_group = 'driver'") is None
    assert validate_sql("WITH d AS (SELECT * FROM lakehouse.gold.monthly_trend) SELECT * FROM d") is None


def test_writes_and_ddl_are_blocked():
    for sql in ("DELETE FROM lakehouse.silver.fact_video",
                "DROP TABLE lakehouse.gold.round_summary",
                "SELECT 1 FROM lakehouse.silver.fact_video; DELETE FROM lakehouse.silver.fact_video",
                "INSERT INTO lakehouse.gold.fix_frequency SELECT * FROM lakehouse.gold.fix_frequency"):
        assert validate_sql(sql) is not None, sql


def test_other_catalogs_are_blocked():
    assert "허용되지 않는 테이블" in validate_sql("SELECT * FROM raw.landing.videos")
    assert "허용되지 않는 테이블" in validate_sql("SELECT * FROM system.runtime.queries")
    assert validate_sql("SELECT * FROM fact_video") is not None  # unqualified names are not allowed


def test_answer_numbers_must_come_from_rows():
    rows = [[0.5222], [91]]
    assert _numbers_ok("드라이버 스트레이트 비율은 52.2%, 샷은 91개예요.", rows)
    assert not _numbers_ok("드라이버 스트레이트 비율은 60%예요.", rows)
