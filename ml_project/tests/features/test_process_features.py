import numpy as np
import pytest

from ml_project.tests.data import df_test
from ml_project.entities import FeatureParams, PreprocessingParams
from ml_project.features import (
    build_transformer,
    make_features,
    get_target
)

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET = "target"


@pytest.fixture
def feature_params() -> FeatureParams:

    params = FeatureParams(
        categorical_features=CATEGORICAL,
        numerical_features=NUMERICAL,
        target_col=TARGET,
    )

    return params


@pytest.fixture
def preprocessing_params() -> PreprocessingParams:

    params = PreprocessingParams(
        scaler="StandardScaler"
    )

    return params


def test_transform_features(df_test, feature_params):

    for scaler in ["StandardScaler", "MinMaxScaler"]:
        preprocessing_params = PreprocessingParams(scaler=scaler)
        transformer = build_transformer(feature_params, preprocessing_params)
        transformer.fit(df_test)
        features = make_features(transformer, df_test)

        numerical_features = features.iloc[:, : len(NUMERICAL)]
        if scaler == "StandardScaler":
            assert np.allclose(numerical_features.mean(axis=0), 0, atol=1e-1), \
                "After StandardScaler mean of scaled features is not equal 0"
            assert np.allclose(numerical_features.std(axis=0), 1, atol=1e-1), \
                "After StandardScaler std of scaled features is not equal 1"
        elif scaler == "MinMaxScaler":
            assert np.all(abs(numerical_features) <= 1), \
                "After MinMaxScaler scaled features are outside of [-1, 1]"

        assert features.isna().sum().sum() == 0, \
            "NaNs are present after transform"
        assert features.shape[0] == df_test.shape[0], \
            "Features must contain the same number of rows as original data"
        assert features.shape[1] > df_test.shape[1], \
            "Categorical features were not one hot encoded"


def test_can_get_target(df_test, feature_params):

    target = get_target(df_test, feature_params)
    target_unique = set(np.unique(target))

    assert target_unique == {0, 1}, f"Target variable contains unexpected values {target_unique}"
