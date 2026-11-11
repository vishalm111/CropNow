"""Main application module for NEST Agricultural Prediction System backend."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.prediction import router as prediction_router
from app.routes.stability import router as stability_router
from app.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ML models on startup so the first request is not slow."""
    logger.info("Loading NEST prediction models …")
    from app.services.inference import ModelRegistry
    registry = ModelRegistry.get()
    logger.info("Models ready: %s", registry.available_models)
    yield
    logger.info("Shutting down NEST backend.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for IoT-Based Agricultural Intelligence",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(stability_router)


@app.get("/")
def get_root():
    """Root endpoint providing service overview."""
    return {
        "project": settings.app_name,
        "description": "IoT-Based Agricultural Intelligence",
        "status": "operational",
        "version": settings.app_version,
        "health_check": "/health",
        "predict_quick": "/predict/quick",
        "predict_batch": "/predict/batch",
        "predict_models": "/predict/models",
        "stability_analyze": "/stability/analyze",
        "documentation": "/docs",
    }
