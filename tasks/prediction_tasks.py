from celery import shared_task
from backend.ml.predictor import ModelPredictor
from backend.services.prediction_service import PredictionService
from backend.services.email_service import EmailService
from backend.core.database import MongoDB
import asyncio

predictor = ModelPredictor()
prediction_service = PredictionService()
email_service = EmailService()

@shared_task(name="process_prediction")
def process_prediction(image_data: str, user_id: str, image_name: str):
    """Async task to process prediction"""
    # This would run in background for large images
    # Implementation would use asyncio.run() to call async functions
    pass

@shared_task(name="generate_report_pdf")
def generate_report_pdf(report_id: str):
    """Async task to generate PDF report"""
    # Implementation for background PDF generation
    pass

@shared_task(name="send_prediction_email")
def send_prediction_email_async(user_email: str, username: str, prediction: str, confidence: float, report_id: str):
    """Send email notification asynchronously"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        email_service.send_prediction_report_email(user_email, username, prediction, confidence, report_id)
    )
    loop.close()

@shared_task(name="cleanup_old_reports")
def cleanup_old_reports(days_old: int = 90):
    """Delete reports older than specified days"""
    # Implementation for cleanup
    pass