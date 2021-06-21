import os

import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    data = data.merge(target, left_index=True, right_index=True)
    train, val = train_test_split(data, test_size=0.25)

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False)


if __name__ == '__main__':
    split()
