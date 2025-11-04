# src/f1kedro/pipelines/regression/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import prepare_regression_features, train_regression_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_regression_features,
                inputs="merged_data",
                outputs="regression_features",
                name="prepare_regression_features_node"
            ),
            node(
                func=train_regression_models,
                inputs="regression_features",
                outputs=["regression_results","regression_models"],
                name="train_regression_models_node"
            ),
        ]
    )
