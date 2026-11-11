"""Health check endpoints for service verification."""
from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return current service health and availability status."""
    return HealthResponse(
        status="healthy",
        service="NEST Agricultural Prediction System Backend",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
