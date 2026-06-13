from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys

# Add the src directory to path to import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.xgboost_model import run_pipeline, test_accuracy, cv_mean, conf_matrix, feat_importances, target_classes, FEATURES, class_dist, top_feature_names, corr_matrix, train_accuracy as train_acc, bal_acc, cv_std

app = FastAPI(title="SmartInfluence API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

class PredictionRequest(BaseModel):
    brand_name: str
    niches: str
    brand_desc: str = ""
    top_n: int = 10

@app.get("/")
def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found."}

@app.get("/api/status")
def status():
    return {
        "status": "online",
        "model_accuracy": test_accuracy,
        "cv_accuracy": cv_mean
    }

@app.get("/api/analysis")
def get_analysis():
    return {
        "features": FEATURES,
        "importances": feat_importances,
        "confusion_matrix": conf_matrix,
        "classes": target_classes,
        "class_distribution": class_dist,
        "top_feature_names": top_feature_names,
        "correlation_matrix": corr_matrix,
        "metrics": {
            "train_accuracy": train_acc,
            "test_accuracy": test_accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "balanced_accuracy": bal_acc
        }
    }

@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        results = run_pipeline(
            brand_name=request.brand_name,
            niches_input=request.niches,
            brand_description=request.brand_desc,
            top_n=request.top_n,
            return_data=True
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
