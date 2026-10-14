"""Nightly lakehouse jobs. The CDC stream keeps silver current during the day; this DAG
verifies and maintains it.

  reconcile        full batch run: re-land every table, MERGE (expect upserted=0 for app
                   tables when the stream is healthy), dbt gold build + tests, DQ checks
  dbt_source_tests dbt tests on the silver sources (keys, relationships, accepted values)
  maintain_<t>     Iceberg upkeep per table: compact small files written by micro-batches,
                   expire old snapshots, remove orphan files
"""

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PY = "/home/airflow/pipeline-venv/bin/python"
DBT = "/home/airflow/pipeline-venv/bin/dbt"
PROJECT = "/opt/swingcoach"
SILVER = ["fact_video", "fact_swing", "fact_card", "dim_reference", "fact_pose_frame",
          "fact_round", "fact_hole", "fact_hole_shot", "fact_driver_landing"]

MAINTAIN = """{py} - <<'EOF'
import sys; sys.path.insert(0, "{project}/pipeline")
from lake import Trino
t = Trino()
fq = "lakehouse.silver.{table}"
t.run(f"ALTER TABLE {{fq}} EXECUTE optimize(file_size_threshold => '64MB')")
t.run(f"ALTER TABLE {{fq}} EXECUTE expire_snapshots(retention_threshold => '7d')")
t.run(f"ALTER TABLE {{fq}} EXECUTE remove_orphan_files(retention_threshold => '7d')")
files = t.scalar('SELECT count(*) FROM lakehouse.silver."{table}$files"')
print(f"{{fq}}: maintained, data files now {{files}}")
EOF"""

with DAG(
    dag_id="swingcoach_lakehouse",
    schedule="0 3 * * *",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["swingcoach"],
) as dag:
    reconcile = BashOperator(task_id="reconcile", bash_command=f"cd {PROJECT} && {PY} pipeline/run.py")

    source_tests = BashOperator(
        task_id="dbt_source_tests",
        bash_command=f"{DBT} test --select source:silver --project-dir {PROJECT}/dbt --profiles-dir {PROJECT}/dbt",
    )

    maintain = [
        BashOperator(task_id=f"maintain_{table}", bash_command=MAINTAIN.format(py=PY, project=PROJECT, table=table))
        for table in SILVER
    ]

    reconcile >> source_tests >> maintain
