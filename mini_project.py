from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Initialize FastAPI app
app = FastAPI(title="House Price Prediction API")

model = joblib.load("regression.joblib")

# First implement a simple GET endpoint
@app.get("/predict")
async def predict_get():
    return {"y_pred": 2}

# Define input model for POST requests - we'll update this later
class HouseData(BaseModel):
    # Placeholder - will be updated when we integrate the model
    features: list[float]

# Home route
@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API"}

# POST endpoint 
@app.post("/predict")
async def predict_post(data: HouseData):
    try:
        features = [data.features]
        prediction = model.predict(features)
        return {"y_pred": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))