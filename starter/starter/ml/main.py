# Put the code for your API here.
from fastapi import FastAPI, Body, Response, status
from pydantic import BaseModel
from .model import load_model, predict_single
import os


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Data(BaseModel):
    workclass: str = None
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlwgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    class Config:
        schema_extra = {
            "example": {
                "workclass": "state_gov",
                "education": "bachelors",
                "marital_status": "never_married",
                "occupation": "adm_clerical",
                "relationship": "not_in_family",
                "race": "white",
                "sex": "male",
                "native_country": "united_states",
                "age": 39,
                "fnlwgt": 77516,
                "education_num": 13,
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40
            }
        }

@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model, encoder, lb = load_model('./model')

#welcome message on the root
@app.get("/")
def read_root():
    response = Response(
        status_code=status.HTTP_200_OK,
        content="Welcome to the Mlops 3 Project"
    )
    return response

#model inference:
@app.post("/predict")
def predict(data: Data):
    #if any data same as example, return error
    #if any(data.dict().values()) == None OR any(data.dict().values()) == "" or any(data.dict().values()) == "string" or any(data.dict().values()) == 0:
    
    #Check if any string data is missing:
    if 'string' in data.dict().values():
        response = Response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content="Please enter all the data correctly"
        )
        return response

    else:

        #predict
        y_pred = predict_single(data, './model')

        response = Response(
            status_code=status.HTTP_200_OK,
            content="The predicted income is: " + str(list(y_pred)[0]),
        )

        return response