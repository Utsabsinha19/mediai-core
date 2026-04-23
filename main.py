from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging

from backend.core.config import settings
from backend.core.database import MongoDB
from backend.routes import auth, prediction, reports, admin, health
from backend.api.middleware import RateLimitMiddleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting up...")
    await MongoDB.connect()
    logger.info("Database connected")
    yield
    # Shutdown
    logger.info("Shutting down...")
    await MongoDB.close()
    logger.info("Database disconnected")

# Create FastAPI app
app = FastAPI(
    title="Healthcare AI Platform API",
    description="AI-powered medical image analysis and disease prediction",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.DEBUG else ["localhost", "127.0.0.1"]
)

# Add rate limiting middleware
if not settings.DEBUG:
    app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(prediction.router, prefix=settings.API_V1_PREFIX)
app.include_router(reports.router, prefix=settings.API_V1_PREFIX)
app.include_router(admin.router, prefix=settings.API_V1_PREFIX)
app.include_router(health.router, prefix=settings.API_V1_PREFIX)

@app.get("/")
async def root():
    return {
        "message": "Healthcare AI Platform API",
        "version": "1.0.0",
        "docs": "/docs" if settings.DEBUG else None,
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )