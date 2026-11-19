from fastapi import APIRouter
from typing import Dict, Any
import psutil
import platform
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "AI_person",
        "version": "0.1.0",
    }


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "status": "healthy",
            "system": {
                "platform": platform.system(),
                "platform_version": platform.version(),
            },
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count(),
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "usage_percent": memory.percent,
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "usage_percent": disk.percent,
            },
        }

    except Exception as e:
        logger.error(f"Get status error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
        }


@router.get("/ready")
async def readiness_check() -> Dict[str, str]:
    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    return {"status": "alive"}
