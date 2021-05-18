import os
import logging
import pickle
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI
from sklearn.pipeline import Pipeline

from entities import InputParams, PredictParams
from models import get_model, make_predict


logger = logging.getLogger(__name__)


model: Optional[Pipeline] = None
    
app = FastAPI()


@app.get("/")
def main() -> str:
    
    return "Hello, it is a first page for your predictions."


@app.on_event("startup")
def load_model() -> Pipeline:
    
    global model
    model_path = os.getenv("PATH_MODEL", "models/model.pkl")
    if model_path is None:
        err = f"PATH_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = get_model(model_path)
    
    return model


@app.get("/health")
def health() -> bool:
    
    return not (model is None)


@app.post("/predict")
def predict(request: InputParams) -> PredictParams:
    
    return make_predict(request, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
