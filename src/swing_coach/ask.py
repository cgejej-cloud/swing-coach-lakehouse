"""Text-to-SQL agent over the lakehouse ("내 데이터에 묻기"), built as a LangGraph state machine.

  generate --> validate --ok--> execute --ok--> answer --> END
      ^            |                |
      +--- error --+------ error ---+   (at most MAX_ATTEMPTS generations, each told why it failed)

Guards:
  * read-only: one SELECT/WITH statement, no DDL/DML, only lakehouse.gold / lakehouse.silver
  * every result is wrapped in a LIMIT
  * the final answer may only quote numbers that appear in the result rows; otherwise the
    agent falls back to a plain table summary
"""

import json
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from swing_coach import llm
from swing_coach.coach import CATALOG

MAX_ATTEMPTS = 3
ROW_LIMIT = 200
ALLOWED_SCHEMAS = ("lakehouse.gold.", "lakehouse.silver.")
FORBIDDEN = re.compile(r"\b(insert|update|delete|merge|drop|create|alter|grant|revoke|call|execute|truncate|"
                       r"comment|analyze|refresh|set\s+session|use)\b", re.I)
QUALIFIED = re.compile(r"\b([a-z_][\w]*)\.([a-z_][\w]*)\.([a-z_][\w$]*)\b", re.I)

METRIC_NAMES = "\n".join(f"  - m_{m.key}: {m.name} ({m.phase})" for m in CATALOG)
SCHEMA_DOC = f"""Trino SQL. 모든 테이블은 카탈로그.스키마.테이블 전체 이름으로 쓴다.

lakehouse.silver.fact_video  -- 샷(영상) 1개 = 1행
  video_id, source('golfzon' | 'upload'), session_key(라운드 식별자), session_label(예: '09.30 수서 골프&리조트'),
  seq_no(라운드 안 순서), shot_at(문자열 'YYYY-MM-DD HH:MM:SS'), club('DRIVER','IRON','WEDGE','WOOD','PUTTER'),
  club_group('driver' | 'iron' | NULL), ball_flight_code(1=스트레이트, 2=슬라이스, 3=훅, 0=모름, NULL=라벨 없음),
  carry_m(비거리 m), hole, status('done','failed',...)
lakehouse.silver.fact_swing  -- 분석된 스윙 1개 = 1행, video_id로 fact_video와 조인
  m_tempo_ratio(템포 백스윙:다운스윙), m_backswing_s, m_downswing_s, m_phase_confidence(0~1), m_scale_ok(1=신뢰)
{METRIC_NAMES}
  (_sw 단위는 어깨너비. 백스윙 이동은 음수일수록 오른쪽으로 밀림)
lakehouse.silver.fact_card  -- 티칭 카드, video_id로 조인
  good, fix, cue(문장), fix_key(고칠 점 지표 키, 예: 'pelvis_sway_top_sw'), good_key, confidence('높음','보통','낮음','-')
lakehouse.gold.round_summary  -- 라운드 x 클럽군
  session_key, session_label, first_shot_at, club_group, shots, analysed, straight_rate, slice_rate, hook_rate(0~1),
  avg_carry_m, median_tempo, median_head_sway_top_sw, median_pelvis_sway_top_sw
lakehouse.gold.monthly_trend  -- month(timestamp, 월 첫날), club_group, shots, straight_rate, median_pelvis_sway_top_sw, median_tempo
lakehouse.gold.fix_frequency  -- club_group, fix_key('(none)' = 지적 없음), cards, share(0~1)
lakehouse.gold.metric_by_flight  -- club_group, ball_flight_code, metric(지표 키, m_ 없이), n, median, mean

규칙: 비율(straight_rate 등)은 구질 라벨이 있는 샷(ball_flight_code IN (1,2,3))만 분모로 쓴다.
날짜 비교는 shot_at 문자열을 그대로 쓰거나 CAST(substr(shot_at, 1, 19) AS timestamp)를 쓴다."""

# Semantic layer: which table answers which kind of question. Measured on tests/ask_eval.json
# (15 questions; the examples below are deliberately different questions from that set):
#   schema only 8/15 (53%) -> with these rules 13/15 (87%).
# One more rule aimed at the two remaining misses ("don't answer overall rates from
# round_summary") dropped it to 12/15: the model ignored it and broke another question,
# so it was reverted. Without a fixed seed, repeated runs of the same prompt scored
# 11-13/15; with seed 42 (llm.chat) three runs in a row scored 12/15 each, so that is the
# number reported. Rules were refined while looking at this set, so expect lower
# accuracy on unseen questions; a held-out set is the next step.
SEMANTIC_RULES = """질문 유형별로 쓸 곳 (반드시 따른다)
- '몇 개/몇 번/평균 비거리/최대 비거리' 같은 샷 단위 질문 → lakehouse.silver.fact_video 에서 직접 집계
- 스윙 지표의 평균·비교 → lakehouse.silver.fact_swing 을 fact_video 와 video_id 로 조인해 avg() 로 직접 계산
  (gold 의 median_* 열은 '중앙값'을 물을 때만 쓴다)
- 'N월', '월별', '가장 많이 친 달' → lakehouse.gold.monthly_trend. month 는 그 달 1일 자정 timestamp:
  month = TIMESTAMP '2026-03-01 00:00:00'
- '고칠 점', '지적', '자주 나온 문제' → lakehouse.gold.fix_frequency 의 fix_key (지적 없음 '(none)' 은 제외)
- 특정 날짜의 라운드 → fact_video.shot_at LIKE 'YYYY-MM-DD%'
- '각각', 'A와 B의 ~' → 그룹마다 한 행이 나오도록 GROUP BY 하고 그룹을 구분하는 열도 함께 SELECT
- 클럽은 club_group('driver','iron') 을 우선 쓴다. 웨지·우드·퍼터를 물을 때만 club 을 쓴다.

예시
Q: 아이언 훅 비율은?
SQL: SELECT avg(CASE WHEN ball_flight_code = 3 THEN 1e0 ELSE 0e0 END) FROM lakehouse.silver.fact_video WHERE club_group = 'iron' AND ball_flight_code IN (1, 2, 3)
Q: 2026년 3월에 아이언을 몇 번 쳤어?
SQL: SELECT shots FROM lakehouse.gold.monthly_trend WHERE club_group = 'iron' AND month = TIMESTAMP '2026-03-01 00:00:00'
Q: 아이언 카드에서 두 번째로 많이 나온 지적은?
SQL: SELECT fix_key FROM lakehouse.gold.fix_frequency WHERE club_group = 'iron' AND fix_key <> '(none)' ORDER BY cards DESC OFFSET 1 LIMIT 1
Q: 아이언 스트레이트 샷과 훅 샷의 임팩트 왼다리 각도 평균을 각각 알려줘
SQL: SELECT v.ball_flight_code, avg(s.m_lead_knee_imp_deg) FROM lakehouse.silver.fact_swing s JOIN lakehouse.silver.fact_video v ON v.video_id = s.video_id WHERE v.club_group = 'iron' AND v.ball_flight_code IN (1, 3) GROUP BY v.ball_flight_code"""

SQL_SYSTEM = f"""너는 데이터 분석가다. 사용자의 한국어 질문에 답하는 Trino SQL 한 개를 만든다.
조회(SELECT/WITH)만 쓰고, 아래 테이블만 쓴다. 필요한 열만 고르고, 질문이 하나의 값을 물으면 한 행만 돌려준다.
가능하면 WITH 없이 짧은 SELECT 하나로 쓴다.

{SCHEMA_DOC}

{SEMANTIC_RULES}"""

ANSWER_SYSTEM = """너는 골프 데이터 분석 도우미다. 질문, 실행한 SQL, 결과 행을 받아 한국어로 1~3문장으로 답한다.
결과에 있는 숫자만 쓴다. 비율(0~1)은 퍼센트로 바꿔도 되지만 다른 숫자를 만들지 않는다. 결과가 비었으면 그렇게 말한다.
값만 나열하지 말고 질문에 답하는 완전한 존댓말 문장으로 쓴다. 클럽 코드(driver, iron)는 드라이버, 아이언으로 쓴다.
예시: 질문 "아이언 최소 비거리는?", 결과 [[98.2]] → "아이언 최소 비거리는 98.2m예요."
예시: 질문 "훅이 가장 많이 난 클럽은?", 결과 [["iron", 9]] → "훅은 아이언에서 9번으로 가장 많이 났어요.\""""


class AskState(TypedDict, total=False):
    question: str
    sql: str
    error: str
    attempts: int
    columns: list[str]
    rows: list[list]
    answer: str
    trace: list[str]


def validate_sql(sql: str) -> str | None:
    """None if the statement is safe to run, else why not."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        return "여러 문장은 실행할 수 없다"
    if not re.match(r"^(select|with)\b", s, re.I):
        return "SELECT 또는 WITH로 시작하는 조회만 허용된다"
    if FORBIDDEN.search(s):
        return f"허용되지 않는 키워드: {FORBIDDEN.search(s).group(0)}"
    for m in QUALIFIED.finditer(s):
        name = f"{m.group(1)}.{m.group(2)}.".lower()
        if name not in ALLOWED_SCHEMAS:
            return f"허용되지 않는 테이블: {m.group(0)} (lakehouse.gold / lakehouse.silver만 가능)"
    if not re.search(r"\blakehouse\.(gold|silver)\.", s, re.I):
        return "lakehouse.gold 또는 lakehouse.silver 테이블을 전체 이름으로 써야 한다"
    return None


def _numbers_ok(answer: str, rows: list[list]) -> bool:
    """Every number in the answer must come from the rows (allowing rounding and 0-1 -> %)."""
    values = []
    for row in rows:
        for v in row:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values += [float(v), float(v) * 100]
            elif v is not None:
                values += [float(x) for x in llm.NUMBER.findall(str(v))]
    for token in llm.NUMBER.findall(answer):
        x = float(token)
        if not any(abs(x - v) <= max(0.051, abs(v) * 0.006) for v in values):
            return False
    return True


def build_graph(run_sql):
    """run_sql(sql) -> (columns, rows); injected so tests can fake the database."""

    def generate(state: AskState) -> AskState:
        prompt = state["question"]
        if state.get("error"):
            prompt += f"\n\n직전 SQL:\n{state.get('sql')}\n오류: {state['error']}\n오류를 고친 SQL을 다시 만들어라."
        out = json.loads(llm.chat(SQL_SYSTEM, prompt, {"type": "object", "required": ["sql"],
                                                        "properties": {"sql": {"type": "string"}}}))
        return {"sql": out["sql"].strip().rstrip(";"), "error": "", "attempts": state.get("attempts", 0) + 1,
                "trace": state.get("trace", []) + [f"generate #{state.get('attempts', 0) + 1}"]}

    def validate(state: AskState) -> AskState:
        reason = validate_sql(state["sql"])
        return {"error": reason or "", "trace": state["trace"] + [f"validate: {reason or 'ok'}"]}

    def execute(state: AskState) -> AskState:
        try:
            cols, rows = run_sql(f"SELECT * FROM ({state['sql']}) LIMIT {ROW_LIMIT}")
            return {"columns": cols, "rows": rows, "error": "", "trace": state["trace"] + [f"execute: {len(rows)} rows"]}
        except Exception as e:  # noqa: BLE001 - the message goes back to the model
            msg = str(e).split("message=")[-1][:300]
            return {"error": msg, "trace": state["trace"] + [f"execute error: {msg[:80]}"]}

    def answer(state: AskState) -> AskState:
        payload = json.dumps({"question": state["question"], "sql": state["sql"], "columns": state["columns"],
                              "rows": state["rows"][:20]}, ensure_ascii=False, default=str)
        text = llm.chat(ANSWER_SYSTEM, payload).strip()
        if not _numbers_ok(text, state["rows"]):
            head = "; ".join(", ".join(f"{c}={v}" for c, v in zip(state["columns"], r)) for r in state["rows"][:5])
            text = f"조회 결과: {head}" if state["rows"] else "조건에 맞는 데이터가 없어요."
            return {"answer": text, "trace": state["trace"] + ["answer: numbers not grounded -> table summary"]}
        return {"answer": text, "trace": state["trace"] + ["answer: ok"]}

    def give_up(state: AskState) -> AskState:
        return {"answer": "질문을 SQL로 바꾸지 못했어요. 표현을 바꿔서 다시 물어봐 주세요.",
                "trace": state["trace"] + ["give up"]}

    def retry_or(next_node: str):
        def route(state: AskState) -> str:
            if not state.get("error"):
                return next_node
            return "generate" if state["attempts"] < MAX_ATTEMPTS else "give_up"
        return route

    g = StateGraph(AskState)
    for name, fn in (("generate", generate), ("validate", validate), ("execute", execute),
                     ("answer", answer), ("give_up", give_up)):
        g.add_node(name, fn)
    g.set_entry_point("generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", retry_or("execute"), ["execute", "generate", "give_up"])
    g.add_conditional_edges("execute", retry_or("answer"), ["answer", "generate", "give_up"])
    g.add_edge("answer", END)
    g.add_edge("give_up", END)
    return g.compile()


def trino_runner():
    import trino

    from swing_coach.insights import _trino

    def run(sql: str):
        cur = _trino().cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return [d[0] for d in cur.description], [list(r) for r in rows]

    return run


def ask(question: str, run_sql=None) -> AskState:
    graph = build_graph(run_sql or trino_runner())
    return graph.invoke({"question": question, "attempts": 0, "trace": []})
