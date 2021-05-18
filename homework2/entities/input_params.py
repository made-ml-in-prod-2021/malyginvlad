from pydantic import BaseModel, Field, validator
import numpy as np


class InputParams(BaseModel):
    age: int = Field(default=13)
    sex: int = Field(default=0)
    cp: int = Field(default=2)
    trestbps: int = Field(default=115)
    chol: int = Field(default=236)
    fbs: int = Field(default=0)
    restecg: int = Field(default=1)
    thalach: int = Field(default=202)
    exang: int = Field(default=0)
    oldpeak: float = Field(default=3.8)
    slope: int = Field(default=1)
    ca: int = Field(default=4)
    thal: int = Field(default=0)
        
    
    @validator('age')
    def check_age(cls, age):
        
        if age < 0:
            raise ValueError(f"The 'age' field must be positive number, but it is {age}")
            
        return age
    
    @validator('sex')
    def check_sex(cls, sex):
        
        if not 0 <= sex < 2:
            raise ValueError(f"The 'sex' field must be 0 or 1, but it is {sex}")
            
        return sex
    
    @validator('oldpeak')
    def check_exang(cls, oldpeak):
        
        if not isinstance(oldpeak, float):
            raise ValueError(f"The 'oldpeak' field must be float, but it is {type(oldpeak)}")
            
        return oldpeak
