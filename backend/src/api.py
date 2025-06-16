from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import numpy as np
import os

app = FastAPI()

W = 0.1119
B = 0.0395

@app.get("/predict/")
async def predict(path: str = Query(..., description="Path to CSV file with 'feature_0' column")):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = pd.read_csv(path)
        if "feature_0" not in df.columns:
            raise HTTPException(status_code=400, detail="Missing 'feature_0' column")

        X = df["feature_0"].values.reshape(-1, 1)
        predictions = (W * X + B).flatten().tolist()

        return {"response": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
