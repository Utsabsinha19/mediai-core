from fastapi import APIRouter, Depends
from datetime import datetime
import platform
import psutil

from backend.core.database import MongoDB

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "healthcare-ai-backend"
    }

@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with system info"""
    # Check database
    db_status = "healthy"
    try:
        await MongoDB.client.admin.command('ping')
    except:
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": {
                "status": db_status,
                "type": "MongoDB"
            },
            "system": {
                "status": "healthy",
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": psutil.cpu_count(),
                "memory_available": psutil.virtual_memory().available,
                "disk_available": psutil.disk_usage('/').free
            }
        }
    }