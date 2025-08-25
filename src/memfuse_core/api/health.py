"""Health API endpoints."""

from fastapi import APIRouter, Depends
from ..models import ApiResponse
from ..utils.config import config_manager
from ..utils.auth import validate_api_key
from ..utils.version import get_version_info


router = APIRouter()


@router.get("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.get("", response_model=ApiResponse)
async def health_check(api_key_data: dict = Depends(validate_api_key)) -> ApiResponse:
    """Check the health of the server."""
    config = config_manager.to_dict()
    version_info = get_version_info()

    return ApiResponse.success(
        data={
            "status": "ok",
            **version_info,  # Expand version info, includes version and python_version
            "config": config
        },
        message="Server is healthy",
    )
