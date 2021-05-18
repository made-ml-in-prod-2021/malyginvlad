import pickle
from typing import Union

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.pipeline import Pipeline

from entities import InputParams, PredictParams


def get_model(path: str) -> Pipeline:

    with open(path, "rb") as file:
        model = pickle.load(file)
    
    return model
    
    
def make_predict(
    request: InputParams,
    model: Pipeline
) -> Union[HTTPException, PredictParams]:
    
    df = pd.DataFrame([dict(request)])
    pred = model.predict(df)
    
    return PredictParams(prediction=pred)
