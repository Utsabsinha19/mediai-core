from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class ReportDownloadRequest(BaseModel):
    report_id: str
    format: str = Field(default="pdf", regex="^(pdf)$")

class ReportSummary(BaseModel):
    id: str
    user_id: str
    username: str
    prediction: str
    confidence: float
    best_model: str
    created_at: datetime
    status: str

class ReportDetailResponse(BaseModel):
    id: str
    user_id: str
    username: str
    email: str
    image_name: str
    prediction: str
    confidence: float
    best_model: str
    all_predictions: Dict[str, List[float]]
    model_metrics: Dict[str, Dict[str, float]]
    shap_plot_base64: Optional[str]
    doctor_notes: Optional[str]
    status: str
    created_at: datetime

class ReportListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    reports: List[ReportSummary]

class ReportUpdateRequest(BaseModel):
    doctor_notes: Optional[str] = None
    status: Optional[str] = Field(None, regex="^(pending|completed|reviewed)$")