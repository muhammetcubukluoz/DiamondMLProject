from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

templates = Jinja2Templates(directory="templates")

app = FastAPI()

with open("30-diamond_model.pkl", "rb") as f:
    save_data = pickle.load(f)
    model = save_data["model"]
    encoders = save_data["encoders"]
    scaler = save_data["scaler"]

class DiamondFeatures(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(features: DiamondFeatures):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([features.model_dump()])

    # Apply label encoding using the saved encoders
    for col in ["cut", "color", "clarity"]:
        input_data[col] = encoders[col].transform(input_data[col])

    # Apply standard scaling using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    return {"predicted_price": prediction[0]}