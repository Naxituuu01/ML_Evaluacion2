# src/f1kedro/pipeline_registry.py
from typing import Dict
from kedro.pipeline import Pipeline
from .pipelines.ingestion.pipeline import create_pipeline as create_ingestion_pipeline
from .pipelines.regression.pipeline import create_pipeline as create_regression_pipeline
from .pipelines.classification.pipeline import create_pipeline as create_classification_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    ingestion = create_ingestion_pipeline()
    regression = create_regression_pipeline()
    classification = create_classification_pipeline()

    # pipeline principal que ejecuta ingestion -> regression + classification
    full = ingestion + regression + classification

    return {
        "__default__": full,
        "ingestion": ingestion,
        "regression": regression,
        "classification": classification,
        "full": full
    }
