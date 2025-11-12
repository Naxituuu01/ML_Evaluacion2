# src/f1kedro/pipelines/regression/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_regression_data, train_regression_models

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_regression_data,
            inputs="merged_data",
            outputs="regression_split_data",
            name="prepare_regression_data_node"
        ),
        node(
            func=train_regression_models,
            inputs="regression_split_data",
            outputs="regression_results",
            name="train_regression_models_node"
        ),
    ])