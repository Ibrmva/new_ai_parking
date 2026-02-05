from fastapi import APIRouter
import httpx
import os
import logging

CAMERA_LIST_URL = os.getenv("CAMERA_LIST_URL")
router = APIRouter()

@router.get("/cameras")
async def get_cameras():
    if not CAMERA_LIST_URL:
        return {"error": "CAMERA_LIST_URL not set"}
    try:
        async with httpx.AsyncClient(timeout=10, verify=False) as client:
            response = await client.get(CAMERA_LIST_URL)
            response.raise_for_status()
            data = response.json()
            return data.get('res', data) if isinstance(data, dict) else data
    except httpx.HTTPError as e:
        logging.error(f"HTTPError: {e}")
        return {"error": f"Failed to fetch cameras: {e}"}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}"}
