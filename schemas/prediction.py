from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class PredictionRequest(BaseModel):
    image_name: str
    image_data: str  # base64 encoded

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence: float

class ModelComparison(BaseModel):
    resnet50: ModelMetrics
    xgboost: ModelMetrics
    random_forest: ModelMetrics
    ensemble: Optional[ModelMetrics]

class PredictionResponse(BaseModel):
    report_id: str
    prediction: str
    confidence: float
    best_model: str
    model_comparison: Dict[str, Dict[str, float]]
    shap_plot: Optional[str] = None
    grad_cam_plot: Optional[str] = None
    created_at: datetime

class PredictionHistoryResponse(BaseModel):
    total: int
    page: int
    per_page: int
    predictions: List[PredictionResponse]

class PredictionUploadResponse(BaseModel):
    success: bool
    message: str
    report_id: Optional[str] = None
    prediction: Optional[PredictionResponse] = None