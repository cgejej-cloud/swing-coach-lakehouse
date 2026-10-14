"""Local LLM (Ollama, Qwen3 8B on the GPU) used in two guarded places:

  * polish_card(): rewrite the rule-based card sentences in a natural coaching tone.
    The rules decide WHAT to say; the model only changes HOW. Any rewrite that adds,
    drops or changes a number, or drifts too far in length, is discarded.
  * ask.py: text-to-SQL over the lakehouse (see there).

Everything degrades to the deterministic path when Ollama is not running.
"""

import json
import os
import re
import urllib.error
import urllib.request

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("SWINGCOACH_LLM_MODEL", "qwen3:8b")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    if os.environ.get("SWINGCOACH_LLM", "on") == "off":
        return False
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def chat(system: str, user: str, schema: dict | None = None, temperature: float = 0.0, timeout: int = 120) -> str:
    """One chat turn. `schema` (JSON Schema) forces structured JSON output."""
    body = {"model": MODEL, "stream": False, "think": False,
            # Fixed seed; GPU kernels can still make runs differ slightly, so evals repeat runs.
            "options": {"temperature": temperature, "seed": 42},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if schema:
        body["format"] = schema
    req = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]["content"]
    except (urllib.error.URLError, OSError) as e:
        raise LLMUnavailable(str(e)) from e


# --- card polishing --------------------------------------------------------------
POLISH_SYSTEM = """너는 스크린골프 연습장에서 옆에 서 있는 친절한 한국인 골프 코치다.
규칙 엔진이 만든 한국어 피드백 문장을 받아서, 뜻은 그대로 두고 말투만 자연스러운 한국어 구어체로 다듬는다.
반드시 지킬 것:
- 출력은 반드시 한국어(한글)로만 쓴다. 일본어·중국어·영어 문장은 절대 쓰지 않는다.
- 숫자와 단위(공 N개, N:1, N° 등)는 하나도 바꾸거나 빼거나 새로 만들지 않는다.
- '기준 스윙보다', '평소 좋은 샷과', '정석 범위' 같은 비교 기준 표현은 원문 그대로 남긴다. 숫자가 무엇과 비교한 값인지 사라지면 안 된다.
- 새로운 진단, 원인, 부상·건강 이야기, 장비 이야기를 추가하지 않는다.
- 각 문장은 원문 길이의 1.5배를 넘지 않는다. 존댓말 한두 문장. 이모지는 넣지 않는다.

예시
입력: {"good": "템포가 좋아요 (3.1:1)", "fix": "임팩트 때 손이 평소보다 뒤에 있어요 (기준 스윙보다 공 2개 정도)", "cue": "다음 샷: \\"손이 공보다 먼저 지나가게\\""}
출력: {"good": "템포 3.1:1, 리듬이 아주 좋아요.", "fix": "임팩트 순간에 손이 기준 스윙보다 공 2개 정도 뒤에 남았어요.", "cue": "다음 샷은 손이 공보다 먼저 지나간다는 느낌으로 쳐 보세요."}
입력: {"good": "임팩트 때 왼다리가 잘 버텨요 (평소 좋은 샷과 같아요)", "fix": "탑에서 머리가 오른쪽으로 밀렸어요 (기준 스윙보다 공 3개 정도)", "cue": "다음 샷: \\"머리는 공 뒤에 두고 어깨만 돌리기\\""}
출력: {"good": "임팩트 때 왼다리가 평소 좋은 샷과 똑같이 잘 버텨 줬어요.", "fix": "탑에서 머리가 기준 스윙보다 공 3개 정도 오른쪽으로 밀렸어요.", "cue": "다음 샷은 머리를 공 뒤에 두고 어깨만 돌려 보세요."}"""

KEEP_PHRASES = ["기준 스윙", "평소 좋은 샷", "정석"]
NON_KOREAN_CJK = re.compile(r"[぀-ヿ一-鿿]")  # kana and CJK ideographs
HANGUL = re.compile(r"[가-힣]")

POLISH_SCHEMA = {"type": "object", "required": ["good", "fix", "cue"],
                 "properties": {"good": {"type": "string"}, "fix": {"type": "string"}, "cue": {"type": "string"}}}


def _strip(text: str | None) -> str:
    return re.sub(r"^[👍👉]\s*", "", text or "").strip()


def _check(original: str, rewritten: str) -> str | None:
    """None if acceptable, else the reason it is rejected."""
    if not original:
        return None if not rewritten else "added text to an empty field"
    if not rewritten:
        return "dropped the sentence"
    # Numbers alone are not enough: a first run kept every number but answered in Japanese.
    if NON_KOREAN_CJK.search(rewritten) or not HANGUL.search(rewritten):
        return "not Korean"
    if sorted(NUMBER.findall(original)) != sorted(NUMBER.findall(rewritten)):
        return f"numbers changed {NUMBER.findall(original)} -> {NUMBER.findall(rewritten)}"
    # A number without what it is compared to changes the meaning.
    for anchor in KEEP_PHRASES:
        if anchor in original and anchor not in rewritten:
            return f"dropped '{anchor}'"
        # Adding one is as bad: "4.1:1 (정석 2.7~3.3)" became "기준 스윙보다 4.1:1".
        if anchor not in original and anchor in rewritten:
            return f"added '{anchor}'"
    if len(rewritten) > 1.6 * len(original) + 10:
        return "too long"
    return None


def polish_card(card) -> tuple[bool, str]:
    """Rewrite card.good / fix / cue in place if every field passes the checks.

    Returns (applied, note). The card is untouched when anything fails.
    """
    fields = {"good": _strip(card.good), "fix": _strip(card.fix), "cue": _strip(card.cue)}
    if not fields["fix"]:
        return False, "nothing to polish"
    prompt = json.dumps(fields, ensure_ascii=False)
    reason = None
    for attempt in range(2):  # one retry, told exactly which rule it broke
        try:
            out = json.loads(chat(POLISH_SYSTEM, prompt, POLISH_SCHEMA))
        except (LLMUnavailable, json.JSONDecodeError, KeyError) as e:
            return False, f"llm error: {type(e).__name__}"
        reason = next((f"{k}: {r}" for k, orig in fields.items()
                       if (r := _check(orig, (out.get(k) or "").strip()))), None)
        if reason is None:
            break
        prompt = (f"{json.dumps(fields, ensure_ascii=False)}\n\n이전 출력이 규칙을 어겼다 ({reason}). "
                  f"원문의 숫자와 비교 기준 표현을 그대로 살려서 다시 써라.")
    if reason:
        return False, f"rejected {reason}"
    card.good = f"👍 {out['good'].strip()}"
    card.fix = f"👉 {out['fix'].strip()}"
    card.cue = out["cue"].strip()
    return True, "polished"
