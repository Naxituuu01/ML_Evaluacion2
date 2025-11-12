from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_classification_data, train_classification_models

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_classification_data,
            inputs="merged_data",
            outputs="classification_split_data",
            name="prepare_classification_data_node"
        ),
        node(
            func=train_classification_models,
            inputs="classification_split_data",
            outputs="classification_results",
            name="train_classification_models_node"
        ),
    ])