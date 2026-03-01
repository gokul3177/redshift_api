from fastapi import FastAPI
import pandas as pd
import joblib
from xgboost import XGBClassifier

app = FastAPI()

print("Loading model...")

# load model
model = XGBClassifier()
model.load_model("redshift_model.json")

# load scaler
scaler = joblib.load("scaler.pkl")

print("Model loaded successfully")

@app.get("/")
def home():
    return {"message": "API working perfectly"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([{
        "alpha": data["alpha"],
        "delta": data["delta"],
        "u": data["u"],
        "g": data["g"],
        "r": data["r"],
        "i": data["i"],
        "z": data["z"],
        "class": data["class"],
        "plate": data["plate"],
        "MJD": data["MJD"]
    }])

    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]

    if pred == 0:
        result = "Local"
    elif pred == 1:
        result = "Medium"
    else:
        result = "Deep Space"

    return {"prediction": result}