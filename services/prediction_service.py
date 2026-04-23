from datetime import datetime
from typing import Dict, Any, List, Optional
from bson import ObjectId

from backend.core.database import MongoDB
from backend.schemas.prediction import PredictionHistoryResponse, PredictionResponse

class PredictionService:
    async def save_report(
        self,
        user_id: str,
        image_name: str,
        prediction_result: Dict[str, Any],
        shap_plot: Optional[str] = None
    ) -> str:
        """Save prediction report to database"""
        report = {
            "user_id": user_id,
            "image_name": image_name,
            "image_path": f"backend/static/uploads/{image_name}",
            "prediction": prediction_result["final_prediction"],
            "confidence": prediction_result["final_confidence"],
            "best_model": prediction_result["best_model"],
            "all_predictions": prediction_result["all_predictions"],
            "model_metrics": prediction_result["model_metrics"],
            "shap_plot_base64": shap_plot,
            "status": "completed",
            "created_at": datetime.utcnow()
        }
        
        collection = MongoDB.get_collection("reports")
        result = await collection.insert_one(report)
        
        return str(result.inserted_id)
    
    async def get_user_reports(
        self,
        user_id: str,
        page: int = 1,
        per_page: int = 10
    ) -> PredictionHistoryResponse:
        """Get user's prediction history"""
        collection = MongoDB.get_collection("reports")
        
        # Get total count
        total = await collection.count_documents({"user_id": user_id})
        
        # Get paginated reports
        skip = (page - 1) * per_page
        cursor = collection.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(per_page)
        
        reports = []
        async for report in cursor:
            reports.append(PredictionResponse(
                report_id=str(report["_id"]),
                prediction=report["prediction"],
                confidence=report["confidence"],
                best_model=report["best_model"],
                model_comparison=report["model_metrics"],
                created_at=report["created_at"]
            ))
        
        return PredictionHistoryResponse(
            total=total,
            page=page,
            per_page=per_page,
            predictions=reports
        )
    
    async def get_report_by_id(self, report_id: str) -> Optional[Dict]:
        """Get report by ID"""
        collection = MongoDB.get_collection("reports")
        try:
            report = await collection.find_one({"_id": ObjectId(report_id)})
            if report:
                report["id"] = str(report["_id"])
            return report
        except:
            return None