from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from IrisModel import IrisModel, IrisFeatures
import numpy as np
import pickle
import uvicorn

app = FastAPI()

# Tentar carregar modelo
try:
    with open('models/iris_model.pkl', 'rb') as model_file:        
        model = IrisModel(pickle.load(model_file))
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found")

@app.post("/predict/")
def predict_iris(features: IrisFeatures):
    print(features)
    try:        
        prediction = model.predict(features)
        return {
            "predicted_class_value": prediction,
            "predicted_class_name": model.class_name_map(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=8000)