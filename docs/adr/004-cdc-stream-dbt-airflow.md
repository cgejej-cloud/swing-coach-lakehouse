# ADR 004 — CDC 스트림(Debezium·Kafka) + dbt gold + Airflow 야간 작업

- 상태: 채택 (2026-10-11)
- 맥락: ADR 001의 전체 스냅샷 배치는 업로드가 레이크에 반영되기까지 배치 주기를 기다린다. 데이터가 커지면 매번 전체를 다시 읽는 비용도 커진다.
  앱 DB가 SQLite라 변경 로그(WAL)를 외부로 흘려보낼 수도 없었다.

## 결정

```
앱 → Postgres(wal_level=logical) → Debezium(pgoutput) → Kafka swingcoach.public.<table>
   → pipeline/stream.py: 마이크로배치(2~5초) · 키별 마지막 상태만 · extract.shape()와 같은 타입 계약
   → raw.cdc.<table> parquet → transform.upsert()로 MERGE / DELETE → dbt gold → Kafka 오프셋 커밋
배치 run.py → 야간 "대조": 전체 스냅샷을 다시 MERGE해 스트림 결과와 차이 0건인지 검증 (+ 포즈 프레임 적재)
Airflow DAG → 대조 → dbt 원천 테스트 → Iceberg 유지보수(optimize, expire_snapshots, remove_orphan_files)
```

| 결정 | 이유 |
|---|---|
| 앱 DB를 Postgres로 이전 (`store.py` 두 백엔드, 작업 큐는 `FOR UPDATE SKIP LOCKED`) | 논리 복제 = 수정일시 없이 모든 변경을 잡는 CDC의 전제 |
| Kafka Connect Iceberg 싱크 대신 직접 만든 소비자 | 배치와 **같은 변환·해시·MERGE 코드**를 쓰게 해 대조로 검증 가능. 싱크 커넥터 빌드·설치 부담도 없음 |
| 오프셋은 레이크 커밋 **후** 커밋 + 키 기준 "마지막 상태" MERGE | 최소 1회 전달 + 멱등 쓰기 = 사실상 정확히 1회. 재전달은 변경 0건 |
| 포즈 프레임은 CDC가 아니라 배치 | 행이 아니라 파일(영상당 약 8천 행). 업로드 후 `run.py --only pose_frame` |
| gold는 dbt가 소유 | 배치·스트림·Airflow가 같은 정의를 쓰고 테스트(19개)가 함께 돈다 |
| Airflow는 별도 가상환경에 파이프라인·dbt 설치 | Airflow 의존성 제약과 충돌 방지 |
| Kafka는 KRaft 단일 노드, 힙 512MB | WSL 7.8GB 안에서 8개 서비스 합계 약 4.8GB |

## 검증 (2026-10-07 ~ 10-11)

| 시나리오 | 결과 |
|---|---|
| 기존 데이터 스냅샷 569 이벤트 처리 | 변경 0건 → 배치와 해시 일치 |
| 앱 API로 기준 스윙 변경 | 2.5초 후 Iceberg 반영 |
| 수정일시 없이 값 변경 / 삽입 / 삭제 | 약 4.5초 후 반영 |
| 스트림 반영 뒤 배치 대조 | 앱 테이블 4개 차이 0건 |
| 레이크 커밋 후 오프셋 커밋 전 장애 → 재시작 | 이벤트 1건 재전달, 변경 0건 |
| Airflow DAG 전체 (`airflow dags test`) | 성공 47초: 대조 차이 0건 · DQ 22/23(경고 1) · dbt 원천 테스트 10/10 · 유지보수 5개 |

## 겪은 문제

- **Kafka 소비자 그룹 재조정**: 장애 후 재시작한 소비자가 아무것도 못 읽음. 죽은 소비자가 세션 타임아웃(기본 45초) 동안 파티션을 쥐고 있었다 → `session.timeout.ms=10000`, 파티션 할당 전에는 유휴로 세지 않음.
- **시계 차이**: WSL VM과 Windows 시계가 약 11초 달라 지연이 −11초로 기록됨 → 같은 시계(Debezium 처리 시각 − Postgres 커밋 시각)로 캡처 지연을 잰다.
- **WSL 유휴 종료**: 붙어 있는 프로세스가 없으면 WSL이 꺼지며 컨테이너도 멈춘다 → `docker compose up`을 붙은 상태로 실행해 둔다(또는 `.wslconfig`의 `vmIdleTimeout` 조정 — 사용자 설정).

## 한계

- 소비자가 단일 프로세스·단일 파티션이라 순서 보장은 쉽지만 확장성은 검증하지 않았다.
- 스키마 변경(열 추가)은 Debezium 이벤트로 들어오면 `ALTER TABLE ADD COLUMN`으로 반영되지만, 타입 변경은 배치 대조의 재구축에 의존한다.
