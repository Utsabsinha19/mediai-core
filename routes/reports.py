from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from typing import Optional
from datetime import datetime
import os

from backend.core.database import MongoDB
from backend.api.dependencies import get_current_user
from backend.utils.pdf_generator import PDFGenerator
from backend.services.report_service import ReportService

router = APIRouter(prefix="/reports", tags=["Reports"])

pdf_generator = PDFGenerator()
report_service = ReportService()

@router.get("/download/{report_id}")
async def download_report(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download report as PDF"""
    # Get report data
    report = await report_service.get_report_by_id(report_id)
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check authorization
    if report["user_id"] != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Generate PDF
    pdf_path = await pdf_generator.generate_medical_report(report, current_user)
    
    # Return file
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"medical_report_{report_id}.pdf"
    )

@router.get("/history")
async def get_report_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get user's report history"""
    reports = await report_service.get_user_reports(
        current_user["id"], 
        page, 
        per_page,
        status
    )
    return reports

@router.get("/analytics")
async def get_report_analytics(
    current_user: dict = Depends(get_current_user)
):
    """Get report analytics for admin"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    analytics = await report_service.get_analytics()
    return analytics