from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "ignacio",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="kedro_f1_pipeline",
    default_args=default_args,
    description="Pipeline de ML para Fórmula 1 - Ingestion, Clasificación y Regresión",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["kedro", "f1", "ml"],
) as dag:

    # 1. INGESTIÓN DE DATOS
    data_ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command="cd /opt/airflow/kedro_project && python -m kedro run --pipeline=ingestion",
    )

    # 2. CLASIFICACIÓN
    classification = BashOperator(
        task_id="classification_pipeline",
        bash_command="cd /opt/airflow/kedro_project && python -m kedro run --pipeline=classification",
    )

    # 3. REGRESIÓN
    regression = BashOperator(
        task_id="regression_pipeline",
        bash_command="cd /opt/airflow/kedro_project && python -m kedro run --pipeline=regression",
    )

    # ORDEN DEL PIPELINE: ingestion -> clasificación -> regresión
    data_ingestion >> classification >> regression