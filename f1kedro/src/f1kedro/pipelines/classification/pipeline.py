# src/f1kedro/pipelines/classification/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import prepare_classification_features, train_classification_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_classification_features,
                inputs="merged_data",
                outputs="classification_features",
                name="prepare_classification_features_node"
            ),
            node(
                func=train_classification_models,
                inputs="classification_features",
                outputs=["classification_results","classification_models"],
                name="train_classification_models_node"
            ),
        ]
    )
