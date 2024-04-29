from fastapi import APIRouter, HTTPException
from src.IrisModel import IrisModel, IrisFeatures
import pickle

router = APIRouter(prefix="/predict")

# Ler modelo treinado
try:
    with open('models/iris_model.pkl', 'rb') as model_file:        
        model = IrisModel(pickle.load(model_file))
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found")

@router.post("/")
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