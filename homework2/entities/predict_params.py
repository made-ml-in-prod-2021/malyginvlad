from pydantic import BaseModel


class PredictParams(BaseModel):
    
    prediction: int
