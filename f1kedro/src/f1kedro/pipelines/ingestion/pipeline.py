# src/f1kedro/pipelines/ingestion/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import merge_datasets

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=merge_datasets,
                inputs=["races_csv","drivers_csv","constructors_csv","results_csv"],
                outputs="merged_data",
                name="merge_raw_data_node",
            ),
        ]
    )
