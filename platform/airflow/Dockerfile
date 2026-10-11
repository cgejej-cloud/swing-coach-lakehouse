# Airflow (Apache-2.0) + an isolated venv for the pipeline and dbt, so their
# dependencies never fight Airflow's own constraints.
FROM apache/airflow:latest-python3.11

USER airflow
RUN python -m venv /home/airflow/pipeline-venv \
 && /home/airflow/pipeline-venv/bin/pip install --no-cache-dir \
      pandas pyarrow boto3 trino "psycopg[binary]" dbt-core dbt-trino
