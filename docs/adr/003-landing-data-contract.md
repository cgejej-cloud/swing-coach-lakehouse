# ADR 003 — 착지 구역 타입 계약과 silver 재구축

- 상태: 채택 (2026-10-07)

## 사건

첫 업로드 영상이 들어온 뒤 자동 동기화가 실패했다. 업로드 영상에는 나스모 번호·구질·홀이 없어(NULL)
pandas가 해당 열을 정수가 아닌 실수로 추론했고, 기존 Iceberg silver 테이블(bigint)과 `MERGE` 타입이 어긋났다.
반대로 처음 적재 때 전부 NULL이던 `club_conf`는 문자열(varchar)로 테이블이 만들어져 있었다.

## 결정

1. **데이터 계약**: `pipeline/extract.py`의 `INT_COLUMNS`, `FLOAT_COLUMNS`로 열 타입을 고정한다. 값이 비어도 타입은 바뀌지 않는다.
2. **타입 드리프트 시 재구축**: silver 열 타입이 계약과 다르면 해당 테이블만 DROP 후 착지 구역 전체 스냅샷에서 다시 적재한다.
   Iceberg는 int→bigint, float→double 확장만 제자리 변경이 되고 varchar→double은 안 된다.
   silver는 파생 계층이고 착지 구역에 전체 스냅샷이 있으므로 재구축이 안전하다.
3. Iceberg에 없는 tinyint/smallint는 integer로 저장되므로 드리프트로 보지 않는다 (그렇지 않으면 매 실행 재구축되어 멱등성이 깨졌다).
4. 새 열(새 지표)은 `ALTER TABLE ADD COLUMN`으로 반영한다 (스키마 진화).

## 검증

- 드리프트 감지 → fact_video, fact_pose_frame 재구축 → 품질 검사 22/22 통과 → 재실행 변경 0건.
- `tests/test_extract_contract.py`: 전부 NULL인 열도 정수/실수 타입을 유지.
