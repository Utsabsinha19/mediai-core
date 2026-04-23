from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from bson import ObjectId

from backend.core.database import MongoDB

class AdminService:
    async def get_all_users(self, page: int = 1, per_page: int = 20, search: Optional[str] = None) -> Dict[str, Any]:
        """Get all users with pagination and search"""
        collection = MongoDB.get_collection("users")
        
        # Build query
        query = {}
        if search:
            query["$or"] = [
                {"username": {"$regex": search, "$options": "i"}},
                {"email": {"$regex": search, "$options": "i"}},
                {"full_name": {"$regex": search, "$options": "i"}}
            ]
        
        # Get total count
        total = await collection.count_documents(query)
        
        # Get paginated users
        skip = (page - 1) * per_page
        cursor = collection.find(query).sort("created_at", -1).skip(skip).limit(per_page)
        
        users = []
        async for user in cursor:
            user["id"] = str(user["_id"])
            user.pop("hashed_password", None)  # Remove password hash
            users.append(user)
        
        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "users": users
        }
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        collection = MongoDB.get_collection("users")
        try:
            user = await collection.find_one({"_id": ObjectId(user_id)})
            if user:
                user["id"] = str(user["_id"])
                user.pop("hashed_password", None)
            return user
        except:
            return None
    
    async def update_user_role(self, user_id: str, role: str) -> bool:
        """Update user role"""
        collection = MongoDB.get_collection("users")
        try:
            result = await collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"role": role, "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except:
            return False
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user and all associated reports"""
        try:
            # Delete user
            users_collection = MongoDB.get_collection("users")
            result = await users_collection.delete_one({"_id": ObjectId(user_id)})
            
            # Delete user's reports
            reports_collection = MongoDB.get_collection("reports")
            await reports_collection.delete_many({"user_id": user_id})
            
            return result.deleted_count > 0
        except:
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        users_collection = MongoDB.get_collection("users")
        reports_collection = MongoDB.get_collection("reports")
        
        # User statistics
        total_users = await users_collection.count_documents({})
        active_users = await users_collection.count_documents({"is_active": True})
        admin_users = await users_collection.count_documents({"role": "admin"})
        
        # User growth (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        new_users_30d = await users_collection.count_documents({"created_at": {"$gte": thirty_days_ago}})
        
        # Report statistics
        total_reports = await reports_collection.count_documents({})
        completed_reports = await reports_collection.count_documents({"status": "completed"})
        pending_reports = await reports_collection.count_documents({"status": "pending"})
        
        # Reports by disease
        pipeline = [
            {"$group": {"_id": "$prediction", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        disease_distribution = []
        async for doc in reports_collection.aggregate(pipeline):
            disease_distribution.append({"disease": doc["_id"], "count": doc["count"]})
        
        # Reports by date (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        pipeline = [
            {"$match": {"created_at": {"$gte": seven_days_ago}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        reports_by_date = []
        async for doc in reports_collection.aggregate(pipeline):
            reports_by_date.append({"date": doc["_id"], "count": doc["count"]})
        
        # Average confidence score
        pipeline = [
            {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence"}}}
        ]
        avg_confidence = 0
        async for doc in reports_collection.aggregate(pipeline):
            avg_confidence = doc["avg_confidence"]
        
        return {
            "users": {
                "total": total_users,
                "active": active_users,
                "admin": admin_users,
                "new_last_30_days": new_users_30d
            },
            "reports": {
                "total": total_reports,
                "completed": completed_reports,
                "pending": pending_reports,
                "avg_confidence": avg_confidence,
                "by_disease": disease_distribution,
                "by_date": reports_by_date
            }
        }
    
    async def get_all_reports(self, page: int = 1, per_page: int = 20, status: Optional[str] = None) -> Dict[str, Any]:
        """Get all reports with pagination and filtering"""
        collection = MongoDB.get_collection("reports")
        
        # Build query
        query = {}
        if status:
            query["status"] = status
        
        # Get total count
        total = await collection.count_documents(query)
        
        # Get paginated reports
        skip = (page - 1) * per_page
        cursor = collection.find(query).sort("created_at", -1).skip(skip).limit(per_page)
        
        reports = []
        async for report in cursor:
            # Get user info
            user = await MongoDB.get_user_by_id(report["user_id"])
            report["id"] = str(report["_id"])
            report["username"] = user["username"] if user else "Unknown"
            reports.append(report)
        
        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "reports": reports
        }
    
    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get aggregated model performance metrics"""
        reports_collection = MongoDB.get_collection("reports")
        
        # Aggregate metrics by model
        pipeline = [
            {"$unwind": "$model_metrics"},
            {"$group": {
                "_id": "$model_metrics",
                "avg_accuracy": {"$avg": "$model_metrics.accuracy"},
                "avg_precision": {"$avg": "$model_metrics.precision"},
                "avg_recall": {"$avg": "$model_metrics.recall"},
                "avg_f1": {"$avg": "$model_metrics.f1_score"},
                "avg_confidence": {"$avg": "$model_metrics.confidence"},
                "count": {"$sum": 1}
            }}
        ]
        
        metrics = {}
        async for doc in reports_collection.aggregate(pipeline):
            metrics[doc["_id"]] = {
                "accuracy": doc["avg_accuracy"],
                "precision": doc["avg_precision"],
                "recall": doc["avg_recall"],
                "f1_score": doc["avg_f1"],
                "confidence": doc["avg_confidence"],
                "sample_count": doc["count"]
            }
        
        # Model usage statistics
        pipeline = [
            {"$group": {"_id": "$best_model", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        model_usage = []
        async for doc in reports_collection.aggregate(pipeline):
            model_usage.append({"model": doc["_id"], "usage_count": doc["count"]})
        
        return {
            "performance_metrics": metrics,
            "model_usage": model_usage,
            "total_predictions": await reports_collection.count_documents({})
        }