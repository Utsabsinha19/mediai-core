from celery import shared_task
from backend.services.email_service import EmailService
import asyncio

email_service = EmailService()

@shared_task(name="send_welcome_email_task")
def send_welcome_email_task(to_email: str, username: str):
    """Send welcome email asynchronously"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(email_service.send_welcome_email(to_email, username))
    loop.close()

@shared_task(name="send_password_reset_email")
def send_password_reset_email(to_email: str, reset_token: str):
    """Send password reset email"""
    subject = "Password Reset Request"
    body = f"""
    <h2>Password Reset Request</h2>
    <p>You requested to reset your password. Click the link below to reset it:</p>
    <p><a href="http://localhost:3000/reset-password?token={reset_token}">Reset Password</a></p>
    <p>This link will expire in 1 hour.</p>
    <p>If you didn't request this, please ignore this email.</p>
    """
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(email_service.send_email(to_email, subject, body, is_html=True))
    loop.close()