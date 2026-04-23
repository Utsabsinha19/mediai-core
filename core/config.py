from decouple import config
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Database
    MONGODB_URL: str = config("MONGODB_URL", default="mongodb://localhost:27017")
    DATABASE_NAME: str = config("DATABASE_NAME", default="healthcare_ai")
    
    # Security
    SECRET_KEY: str = config("SECRET_KEY", default="your-secret-key")
    ALGORITHM: str = config("ALGORITHM", default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30, cast=int)
    REFRESH_TOKEN_EXPIRE_DAYS: int = config("REFRESH_TOKEN_EXPIRE_DAYS", default=7, cast=int)
    
    # Redis
    REDIS_URL: str = config("REDIS_URL", default="redis://localhost:6379")
    CELERY_BROKER_URL: str = config("CELERY_BROKER_URL", default="redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = config("CELERY_RESULT_BACKEND", default="redis://localhost:6379/1")
    
    # Email
    SMTP_HOST: str = config("SMTP_HOST", default="")
    SMTP_PORT: int = config("SMTP_PORT", default=587, cast=int)
    SMTP_USER: str = config("SMTP_USER", default="")
    SMTP_PASSWORD: str = config("SMTP_PASSWORD", default="")
    EMAIL_FROM: str = config("EMAIL_FROM", default="noreply@healthcare-ai.com")
    
    # File Upload
    MAX_FILE_SIZE: int = config("MAX_FILE_SIZE", default=10485760, cast=int)
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".dcm"]
    
    # Model Paths
    RESNET_MODEL_PATH: str = config("RESNET_MODEL_PATH", default="backend/ml/weights/resnet50_best.pth")
    XGBOOST_MODEL_PATH: str = config("XGBOOST_MODEL_PATH", default="backend/ml/weights/xgboost_model.pkl")
    RF_MODEL_PATH: str = config("RF_MODEL_PATH", default="backend/ml/weights/random_forest.pkl")
    ENSEMBLE_MODEL_PATH: str = config("ENSEMBLE_MODEL_PATH", default="backend/ml/weights/ensemble_model.pkl")
    
    # CORS
    FRONTEND_URL: str = config("FRONTEND_URL", default="http://localhost:5173")
    BACKEND_URL: str = config("BACKEND_URL", default="http://localhost:8000")
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        return [self.FRONTEND_URL, self.BACKEND_URL]
    
    # API
    API_V1_PREFIX: str = config("API_V1_PREFIX", default="/api/v1")
    RATE_LIMIT_PER_MINUTE: int = config("RATE_LIMIT_PER_MINUTE", default=60, cast=int)
    RATE_LIMIT_PER_HOUR: int = config("RATE_LIMIT_PER_HOUR", default=1000, cast=int)
    
    # Logging
    LOG_LEVEL: str = config("LOG_LEVEL", default="INFO")
    LOG_FILE: str = config("LOG_FILE", default="logs/app.log")
    
    # Environment
    ENVIRONMENT: str = config("ENVIRONMENT", default="development")
    DEBUG: bool = config("DEBUG", default=True, cast=bool)
    
    # Disease classes
    DISEASE_CLASSES: List[str] = [
        "Glioma",
        "Meningioma", 
        "Pituitary",
        "Normal"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()