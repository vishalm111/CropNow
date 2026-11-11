"""Application settings and configuration."""
import os
from functools import lru_cache
from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "NEST Agricultural Prediction System"
    app_version: str = "0.1.0"
    debug: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
