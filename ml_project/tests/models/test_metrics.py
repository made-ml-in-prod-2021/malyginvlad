import os

from ml_project.train_pipeline import train_pipeline
from ml_project.entities import read_training_pipeline_params

CONFIG_PATH = "ml_project/configs/test_config.yaml"


def test_train_metrics():

    params = read_training_pipeline_params(CONFIG_PATH)
    metrics = train_pipeline(params)

    assert {"precision", "recall", "f1_score"} == metrics.keys(), f"Unexpected metrics output {metrics.keys()}"
    assert metrics["f1_score"] > .5, f"F1 score {metrics['f1_score']} < 0.5"
    assert os.path.isfile(params.output_model_path), f"No such file {params.output_model_path}"
    assert os.path.isfile(params.metric_path), f"No such file {params.metric_path}"

