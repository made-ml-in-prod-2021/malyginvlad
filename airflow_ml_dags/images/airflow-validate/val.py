import os
import pickle
import json

import pandas as pd
import click
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


@click.command("validate")
@click.option("--data-dir")
@click.option("--model-dir")
def val(data_dir: str, model_dir: str):

    with open(os.path.join(model_dir, "logreg.pkl"), "rb") as f:
        model = pickle.load(f)
        
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
        
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))

    val_X, val_y = val.drop('target', axis=1), val['target']
    val_X = scaler.transform(val_X)
    preds = model.predict(val_X)
    metrics = {
        "roc_auc_score": roc_auc_score(val_y, preds),
        "accuracy_score": accuracy_score(val_y, preds),
        "f1_score": f1_score(val_y, preds),
    }

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    val()
