import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from backend.core.config import settings

class EmailService:
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.email_from = settings.EMAIL_FROM
        
        self.enabled = bool(self.smtp_host and self.smtp_user and self.smtp_password)
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        is_html: bool = False
    ) -> bool:
        """Send email to recipient"""
        if not self.enabled:
            print(f"Email service disabled. Would send: {subject} to {to_email}")
            return False
        
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_from
            msg["To"] = to_email
            msg["Subject"] = subject
            
            if is_html:
                msg.attach(MIMEText(body, "html"))
            else:
                msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False
    
    async def send_welcome_email(self, to_email: str, username: str) -> bool:
        """Send welcome email to new user"""
        subject = "Welcome to Healthcare AI Platform"
        body = f"""
        <h2>Welcome {username}!</h2>
        <p>Thank you for joining Healthcare AI Platform.</p>
        <p>You can now:</p>
        <ul>
            <li>Upload medical images for AI analysis</li>
            <li>Get instant disease predictions</li>
            <li>Download detailed medical reports</li>
            <li>Track your health history</li>
        </ul>
        <p>If you have any questions, please contact our support team.</p>
        <br>
        <p>Best regards,<br>Healthcare AI Team</p>
        """
        return await self.send_email(to_email, subject, body, is_html=True)
    
    async def send_prediction_report_email(
        self,
        to_email: str,
        username: str,
        prediction: str,
        confidence: float,
        report_id: str
    ) -> bool:
        """Send email notification with prediction results"""
        subject = f"Your Medical Report is Ready - {prediction}"
        body = f"""
        <h2>Hello {username},</h2>
        <p>Your medical image analysis is complete.</p>
        <h3>Results Summary:</h3>
        <ul>
            <li><strong>Prediction:</strong> {prediction}</li>
            <li><strong>Confidence:</strong> {confidence:.1%}</li>
            <li><strong>Report ID:</strong> {report_id}</li>
        </ul>
        <p>Please log in to your account to view and download the full report.</p>
        <p><strong>Important:</strong> This is an AI-generated report. Please consult with a healthcare professional for medical advice.</p>
        <br>
        <p>Best regards,<br>Healthcare AI Team</p>
        """
        return await self.send_email(to_email, subject, body, is_html=True)