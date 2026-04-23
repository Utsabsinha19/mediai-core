import os
import shutil
import uuid
from typing import Optional
from pathlib import Path

class StorageService:
    def __init__(self, base_path: str = "backend/static"):
        self.base_path = Path(base_path)
        self.uploads_path = self.base_path / "uploads"
        self.reports_path = self.base_path / "reports"
        self.temp_path = self.base_path / "temp"
        
        # Create directories
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_file(self, file_content: bytes, original_filename: str) -> str:
        """Save uploaded file and return path"""
        # Generate unique filename
        ext = Path(original_filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = self.uploads_path / unique_filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return str(file_path)
    
    async def save_report_pdf(self, pdf_content: bytes, report_id: str) -> str:
        """Save generated PDF report"""
        filename = f"report_{report_id}.pdf"
        file_path = self.reports_path / filename
        
        with open(file_path, "wb") as f:
            f.write(pdf_content)
        
        return str(file_path)
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception:
            return False
    
    async def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except:
            return 0
    
    async def cleanup_temp_files(self, age_hours: int = 24):
        """Clean up temporary files older than specified hours"""
        import time
        current_time = time.time()
        
        for file_path in self.temp_path.iterdir():
            if file_path.is_file():
                file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
                if file_age_hours > age_hours:
                    file_path.unlink()