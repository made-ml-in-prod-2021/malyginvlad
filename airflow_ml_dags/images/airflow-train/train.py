import os
import pickle

import pandas as pd
import numpy as np
import click
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--data-dir")
@click.option("--model-dir")
def train(data_dir: str, model_dir: str):

    X_train = pd.read_csv(os.path.join(data_dir, "train_scaled.csv"), header=None)
    y_train = pd.read_csv(os.path.join(data_dir, "train_target.csv"))
    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "logreg.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
