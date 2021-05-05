import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from ml_project.entities.train_params import TrainingParams


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_params: TrainingParams
) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_estimators=train_params.n_estimators,
        criterion=train_params.criterion,
        max_depth=train_params.max_depth,
        random_state=train_params.random_state
    )
    model.fit(features, target)

    return model


def predict_model(
    model: RandomForestClassifier,
    features: pd.DataFrame
) -> np.ndarray:

    predicted = model.predict(features)

    return predicted


def evaluate_model(
    predicted: np.ndarray,
    target: pd.Series
) -> Dict[str, float]:

    scores = {
        "precision": precision_score(target, predicted),
        "recall": recall_score(target, predicted),
        "f1_score": f1_score(target, predicted),
    }

    return scores


def save_model(
    model: RandomForestClassifier,
    output: str
) -> str:

    with open(output, "wb") as file:
        pickle.dump(model, file)

    return output
