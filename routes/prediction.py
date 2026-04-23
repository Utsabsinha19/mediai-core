from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, status
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List
import os
import uuid
from bson import ObjectId

from backend.core.database import MongoDB
from backend.core.config import settings
from backend.api.dependencies import get_current_user
from backend.ml.predictor import ModelPredictor
from backend.utils.image_processor import ImageProcessor
from backend.utils.shap_explainer import SHAPExplainer
from backend.services.prediction_service import PredictionService
from backend.schemas.prediction import PredictionResponse, PredictionHistoryResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Initialize components
image_processor = ImageProcessor()
predictor = ModelPredictor()
shap_explainer = SHAPExplainer()
prediction_service = PredictionService()

@router.post("/upload")
async def predict_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload and predict medical image"""
    try:
        # Validate file
        if not image_processor.validate_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Read and process image
        contents = await file.read()
        if len(contents) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Process image
        processed_image = await image_processor.process_image(contents, file.filename)
        
        # Run prediction
        prediction_result = await predictor.predict(processed_image)
        
        # Generate SHAP explanation
        shap_plot = await shap_explainer.explain(prediction_result)
        
        # Save report to database
        report_id = await prediction_service.save_report(
            user_id=current_user["id"],
            image_name=file.filename,
            prediction_result=prediction_result,
            shap_plot=shap_plot
        )
        
        # Prepare response
        response = PredictionResponse(
            report_id=report_id,
            prediction=prediction_result["final_prediction"],
            confidence=prediction_result["final_confidence"],
            best_model=prediction_result["best_model"],
            model_comparison=prediction_result["model_metrics"],
            shap_plot=shap_plot,
            created_at=datetime.utcnow()
        )
        
        return JSONResponse(content=response.dict(), status_code=200)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{user_id}", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    user_id: str,
    page: int = 1,
    per_page: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get user's prediction history"""
    # Check authorization
    if current_user["id"] != user_id and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    reports = await prediction_service.get_user_reports(user_id, page, per_page)
    return reports

@router.get("/report/{report_id}")
async def get_report(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get specific report"""
    report = await prediction_service.get_report_by_id(report_id)
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check authorization
    if report["user_id"] != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return report