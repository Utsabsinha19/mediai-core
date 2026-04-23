from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
from bson import ObjectId

from backend.core.database import MongoDB
from backend.api.dependencies import get_current_admin, get_current_user
from backend.models.user import UserResponse
from backend.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["Admin"])
admin_service = AdminService()

@router.get("/users")
async def get_all_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    current_user: dict = Depends(get_current_admin)
):
    """Get all users (admin only)"""
    users = await admin_service.get_all_users(page, per_page, search)
    return users

@router.get("/users/{user_id}")
async def get_user_details(
    user_id: str,
    current_user: dict = Depends(get_current_admin)
):
    """Get user details (admin only)"""
    user = await admin_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role: str,
    current_user: dict = Depends(get_current_admin)
):
    """Update user role (admin only)"""
    if role not in ["user", "admin"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    result = await admin_service.update_user_role(user_id, role)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User role updated successfully"}

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: dict = Depends(get_current_admin)
):
    """Delete user (admin only)"""
    if user_id == current_user["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    result = await admin_service.delete_user(user_id)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User deleted successfully"}

@router.get("/stats")
async def get_system_stats(
    current_user: dict = Depends(get_current_admin)
):
    """Get system statistics (admin only)"""
    stats = await admin_service.get_system_stats()
    return stats

@router.get("/reports")
async def get_all_reports(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_admin)
):
    """Get all reports (admin only)"""
    reports = await admin_service.get_all_reports(page, per_page, status)
    return reports

@router.get("/model-metrics")
async def get_model_metrics(
    current_user: dict = Depends(get_current_admin)
):
    """Get model performance metrics (admin only)"""
    metrics = await admin_service.get_model_metrics()
    return metrics