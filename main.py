from fastapi import FastAPI
import pandas as pd
import joblib
from xgboost import XGBClassifier
import os
import requests

app = FastAPI()

# ---------- DOWNLOAD MODEL ----------
MODEL_URL = "https://huggingface.co/gokul3177/redshift-model/resolve/main/redshift_model.json"
SCALER_URL = "https://huggingface.co/gokul3177/redshift-model/resolve/main/scaler.pkl"

if not os.path.exists("redshift_model.json"):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    open("redshift_model.json", "wb").write(r.content)

if not os.path.exists("scaler.pkl"):
    print("Downloading scaler...")
    r = requests.get(SCALER_URL)
    open("scaler.pkl", "wb").write(r.content)

print("Loading model...")

model = XGBClassifier()
model.load_model("redshift_model.json")

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

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)