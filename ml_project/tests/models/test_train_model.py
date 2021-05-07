import pickle
from typing import Tuple
from pathlib import Path

import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml_project.tests.data import df_test
from ml_project.tests.features import feature_params, preprocessing_params
from ml_project.features import (
    build_transformer,
    make_features,
    get_target
)
from ml_project.entities import TrainingParams, FeatureParams, PreprocessingParams
from ml_project.models.model_train import (
    train_model,
    predict_model,
    save_model
)


@pytest.fixture
def get_features_and_target(
    df_test: pd.DataFrame,
    feature_params: FeatureParams,
    preprocessing_params: PreprocessingParams
) -> Tuple[pd.DataFrame, pd.Series]:

    transformer = build_transformer(feature_params, preprocessing_params)
    transformer.fit(df_test)
    features = make_features(transformer, df_test)
    target = get_target(df_test, feature_params)

    return features, target


def test_train_model(get_features_and_target: Tuple[pd.DataFrame, pd.Series]):

    features, target = get_features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    predicted_shape = predict_model(model, features).shape[0]

    assert isinstance(model, RandomForestClassifier), f"Unexpected model {model.__dict__['base_estimator']}"
    assert predicted_shape == target.shape[0], f"Predicted shape {predicted_shape} while should be {target.shape[0]}"


def test_save_model(tmpdir: Path):

    model = RandomForestClassifier(n_estimators=50)
    expected_path = tmpdir.join("model.pkl")
    save_model(model, expected_path)
    with open(expected_path, "rb") as f:
        model = pickle.load(f)

    assert isinstance(model, RandomForestClassifier), f"Unexpected model {model.__dict__['base_estimator']}"




