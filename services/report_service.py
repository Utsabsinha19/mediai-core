from datetime import datetime
from typing import Dict, Any, List, Optional
from bson import ObjectId

from backend.core.database import MongoDB

class ReportService:
    async def get_report_by_id(self, report_id: str) -> Optional[Dict]:
        """Get report by ID"""
        collection = MongoDB.get_collection("reports")
        try:
            report = await collection.find_one({"_id": ObjectId(report_id)})
            if report:
                report["id"] = str(report["_id"])
                report["_id"] = str(report["_id"])
            return report
        except Exception as e:
            print(f"Error getting report: {e}")
            return None
    
    async def get_user_reports(
        self,
        user_id: str,
        page: int = 1,
        per_page: int = 10,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user's reports with pagination"""
        collection = MongoDB.get_collection("reports")
        
        # Build query
        query = {"user_id": user_id}
        if status:
            query["status"] = status
        
        # Get total count
        total = await collection.count_documents(query)
        
        # Get paginated reports
        skip = (page - 1) * per_page
        cursor = collection.find(query).sort("created_at", -1).skip(skip).limit(per_page)
        
        reports = []
        async for report in cursor:
            reports.append({
                "id": str(report["_id"]),
                "user_id": report["user_id"],
                "image_name": report["image_name"],
                "prediction": report["prediction"],
                "confidence": report["confidence"],
                "best_model": report["best_model"],
                "status": report["status"],
                "created_at": report["created_at"].isoformat()
            })
        
        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "reports": reports
        }
    
    async def update_report_status(self, report_id: str, status: str, doctor_notes: Optional[str] = None) -> bool:
        """Update report status and add doctor notes"""
        collection = MongoDB.get_collection("reports")
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            if doctor_notes:
                update_data["doctor_notes"] = doctor_notes
            
            result = await collection.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except:
            return False
    
    async def delete_report(self, report_id: str) -> bool:
        """Delete a report"""
        collection = MongoDB.get_collection("reports")
        try:
            result = await collection.delete_one({"_id": ObjectId(report_id)})
            return result.deleted_count > 0
        except:
            return False
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get report analytics for dashboard"""
        reports_collection = MongoDB.get_collection("reports")
        
        # Total reports by status
        total_reports = await reports_collection.count_documents({})
        completed = await reports_collection.count_documents({"status": "completed"})
        pending = await reports_collection.count_documents({"status": "pending"})
        reviewed = await reports_collection.count_documents({"status": "reviewed"})
        
        # Average confidence by disease
        pipeline = [
            {"$group": {
                "_id": "$prediction",
                "avg_confidence": {"$avg": "$confidence"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        
        disease_stats = []
        async for doc in reports_collection.aggregate(pipeline):
            disease_stats.append({
                "disease": doc["_id"],
                "avg_confidence": doc["avg_confidence"],
                "count": doc["count"]
            })
        
        # Reports trend (last 30 days)
        thirty_days_ago = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        pipeline = [
            {"$match": {"created_at": {"$gte": thirty_days_ago}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        trend = []
        async for doc in reports_collection.aggregate(pipeline):
            trend.append({"date": doc["_id"], "reports": doc["count"]})
        
        return {
            "summary": {
                "total_reports": total_reports,
                "completed": completed,
                "pending": pending,
                "reviewed": reviewed
            },
            "disease_distribution": disease_stats,
            "reports_trend": trend,
            "completion_rate": (completed / total_reports * 100) if total_reports > 0 else 0
        }