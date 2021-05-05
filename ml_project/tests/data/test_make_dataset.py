import numpy as np
import pandas as pd
import pytest

from ml_project.data.make_dataset import read_data, split_train_test_data
from ml_project.entities import SplittingParams

TRAIN_DATASET_PATH = "ml_project/data/raw/heart.csv"
TEST_DATASET_PATH = "ml_project/tests/data/test_dataset.csv"
TEST_SIZE = 300


@pytest.fixture()
def df_test() -> pd.DataFrame:
    
    df_train = read_data()
    columns = list(df_train.columns)
    test_data = {
        col: np.random.choice(df_train[col].unique(), size=TEST_SIZE) for col in columns
    }
    df_test = pd.DataFrame(test_data)
    df_test.to_csv(TEST_DATASET_PATH, index=False)

    return df_test


def test_load_dataset(df_test):

    df_train = read_data()

    assert df_test.shape[0] == TEST_SIZE
    assert df_test.isna().sum().sum() >= 0, "Test data does not have NaNs"
    assert list(df_test.columns).sort() == list(df_train.columns).sort(), \
        "Test columns and train columns are not equal"


def test_split_dataset(df_test):

    size_split = 0.3
    splitting_params = SplittingParams(random_state=13, test_size=size_split)
    train, test = split_train_test_data(df_test, splitting_params)

    assert train.shape[0] == TEST_SIZE * (1 - size_split), "Wrong train shape"
    assert test.shape[0] == TEST_SIZE * size_split, "Wrong test shape"
