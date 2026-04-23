from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from bson import ObjectId

class Report(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str = Field(...)
    image_name: str = Field(...)
    image_path: str = Field(...)
    prediction: str = Field(...)
    confidence: float = Field(..., ge=0, le=1)
    best_model: str = Field(...)
    all_predictions: Dict[str, Any] = Field(default_factory=dict)
    model_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    shap_plot_base64: Optional[str] = None
    grad_cam_base64: Optional[str] = None
    status: str = Field(default="pending")  # pending, completed, reviewed
    doctor_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}

class ReportCreate(BaseModel):
    image_name: str
    image_path: str
    prediction: str
    confidence: float
    best_model: str
    all_predictions: Dict[str, Any]
    model_metrics: Dict[str, Dict[str, float]]
    shap_plot_base64: Optional[str] = None

class ReportResponse(BaseModel):
    id: str
    user_id: str
    image_name: str
    prediction: str
    confidence: float
    best_model: str
    model_metrics: Dict[str, Dict[str, float]]
    status: str
    doctor_notes: Optional[str]
    created_at: datetime
    report_url: Optional[str] = None

class ReportHistoryResponse(BaseModel):
    total: int
    page: int
    per_page: int
    reports: List[ReportResponse]