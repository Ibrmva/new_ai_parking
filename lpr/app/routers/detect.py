# from fastapi import APIRouter, UploadFile, File, Form, Path, HTTPException
# import os
# import numpy as np
# import cv2
# from datetime import datetime, timezone, timedelta
# import base64
# import logging
# import httpx
# from sqlalchemy.future import select

# from lpr.app.ml_utils import ml_service
# from lpr.app.models import Plate
# from lpr.app.database import async_session

# # Environment variables
# PLATE_RECOGNITION_ADD_URL = os.getenv("PLATE_RECOGNITION_ADD_URL")
# PLATE_IMAGE_URL = os.getenv("PLATE_IMAGE_URL")
# EXTERNAL_RECOGNITION_URL = os.getenv("EXTERNAL_RECOGNITION_URL")

# router = APIRouter()

# logger = logging.getLogger(__name__)


# def build_detection_payload(
#     plate_id: int,
#     filename: str,
#     detected_text: str,
#     confidence: float,
#     created_at: datetime,
#     camera_id: str,
#     bbox: list,
#     image_bytes: bytes
# ) -> dict:
#     """
#     Build the detection payload matching the required JSON schema.
    
#     Required schema:
#     {
#       "id": 0,
#       "filename": "string",
#       "detectedText": "string",
#       "confidence": 0.1,
#       "createdAt": "2026-01-30T09:56:35.518Z",
#       "cameraId": 0,
#       "bbox": "string",
#       "image": "string"
#     }
#     """
#     # Format bbox as comma-separated string
#     bbox_str = ",".join(map(str, bbox)) if bbox else ""
    
#     # Format created_at in ISO format with Z suffix
#     if created_at.tzinfo is None:
#         created_at = created_at.replace(tzinfo=timezone(timedelta(hours=6)))
#     created_at_iso = created_at.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    
#     # Convert image bytes to base64 without changing format
#     image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
#     return {
#         "id": int(plate_id),
#         "filename": filename,
#         "detectedText": detected_text,
#         "confidence": float(confidence),
#         "createdAt": created_at_iso,
#         "cameraId": int(camera_id) if camera_id.isdigit() else camera_id,
#         "bbox": bbox_str,
#         "image": image_base64
#     }


# async def send_to_external_api(plate_id: int, payload: dict) -> dict:
#     """
#     Send detection result to external API.
#     """
#     if not PLATE_RECOGNITION_ADD_URL:
#         logger.warning("PLATE_RECOGNITION_ADD_URL not configured")
#         return {"success": False, "error": "External URL not configured", "plateId": plate_id}
    
#     try:
#         async with httpx.AsyncClient(timeout=60) as client:
#             response = await client.post(PLATE_RECOGNITION_ADD_URL, json=payload)
#             response.raise_for_status()
#             logger.info(f"Successfully sent plate {plate_id} to external API")
#             return {"success": True, "data": response.json(), "plateId": plate_id}
            
#     except httpx.HTTPStatusError as e:
#         logger.error(f"External API error for plate {plate_id}: {e.response.status_code} - {e.response.text}")
#         return {"success": False, "error": "HTTP error", "plateId": plate_id, "details": e.response.text}
#     except httpx.RequestError as e:
#         logger.error(f"Request to external API for plate {plate_id} failed: {e}")
#         return {"success": False, "error": "Connection failed", "plateId": plate_id}


# async def process_image_detection(image_bytes: bytes, filename: str, camera_id: str = "0"):
#     """
#     Process image for license plate detection and recognition.
    
#     Args:
#         image_bytes: Raw image bytes (original format preserved)
#         filename: Original filename
#         camera_id: Camera identifier
    
#     Returns:
#         List of detection results
#     """
#     # Decode image
#     np_img = np.frombuffer(image_bytes, np.uint8)
#     image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
#     if image is None:
#         return [{"error": "Could not decode image", "filename": filename}]
    
#     # Determine image format from filename or bytes
#     original_format = 'JPEG'
#     if filename.lower().endswith('.png'):
#         original_format = 'PNG'
#     elif filename.lower().endswith('.bmp'):
#         original_format = 'BMP'
#     elif filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
#         original_format = 'TIFF'
    
#     # Save temp file for model inference
#     temp_path = f"temp_process_{datetime.now().timestamp()}.jpg"
#     try:
#         # Save as JPEG for YOLO model (it requires a file)
#         cv2.imwrite(temp_path, image)
        
#         # Detect plates
#         detected_plates = ml_service.detect_plates(temp_path, original_image_bytes=image_bytes)
        
#     finally:
#         if os.path.exists(temp_path):
#             os.unlink(temp_path)
    
#     results = []
    
#     async with async_session() as session:
#         for plate_info in detected_plates:
#             cropped_image = plate_info["cropped_image"]
#             bbox = plate_info.get("bbox", [])
            
#             # Recognize text
#             recognized_text, (_, confidences) = ml_service.recognize("lpr", cropped_image)
            
#             # Calculate average confidence
#             if confidences:
#                 avg_conf = float(np.mean([float(c) for c in confidences]))
#             else:
#                 avg_conf = plate_info.get("confidence", 0.0)
            
#             # Get track_id
#             last_track_query = await session.execute(select(Plate.track_id).order_by(Plate.track_id.desc()))
#             last_track = last_track_query.scalars().first() or 0
#             track_id = last_track + 1
            
#             # Encode cropped plate image - preserve original format
#             # We use the original image bytes for the full image
#             plate_image_bytes = cv2.imencode('.jpg', cropped_image)[1].tobytes()
            
#             # Create timestamp
#             created_at = datetime.now(timezone(timedelta(hours=6)))
            
#             # Create database record
#             new_plate = Plate(
#                 filename=filename,
#                 detected_text=recognized_text,
#                 confidence=str(avg_conf),
#                 camera_id=camera_id,
#                 image=plate_image_bytes,  # Store cropped plate
#                 track_id=track_id,
#                 image_id=int(0),
#                 created_at=created_at,
#                 bbox=",".join(map(str, bbox)) if bbox else None
#             )
#             session.add(new_plate)
#             await session.commit()
#             await session.refresh(new_plate)
            
#             # Update image_id with plate id
#             new_plate.image_id = new_plate.id
#             await session.commit()
            
#             # Build payload matching required JSON schema
#             payload = build_detection_payload(
#                 plate_id=new_plate.id,
#                 filename=filename,
#                 detected_text=recognized_text,
#                 confidence=avg_conf,
#                 created_at=created_at,
#                 camera_id=camera_id,
#                 bbox=bbox,
#                 image_bytes=image_bytes  # Send full original image, not cropped
#             )
            
#             # Send to external API
#             external_result = await send_to_external_api(new_plate.id, payload)
#             payload["externalRecognition"] = external_result
            
#             results.append(payload)
    
#     return results


# @router.post("/process_image")
# async def process_image(camera_id: str = Form("0"), file: UploadFile = File(...)):
#     """
#     Process uploaded image for license plate detection and recognition.
    
#     - **camera_id**: Camera identifier
#     - **file**: Image file to process
#     """
#     contents = await file.read()
    
#     if not contents:
#         raise HTTPException(status_code=400, detail="Empty file")
    
#     processed_results = await process_image_detection(
#         image_bytes=contents,
#         filename=file.filename or "uploaded_image.jpg",
#         camera_id=camera_id
#     )
    
#     return {"processed_plates": processed_results}


# @router.post("/process_image_url")
# async def process_image_url(camera_id: str = Form("1")):
#     """
#     Process image from configured URL (PLATE_IMAGE_URL).
    
#     - **camera_id**: Camera identifier
#     """
#     if not PLATE_IMAGE_URL:
#         raise HTTPException(status_code=500, detail="PLATE_IMAGE_URL is not set in environment variables")
    
#     try:
#         async with httpx.AsyncClient(timeout=30) as client:
#             response = await client.get(PLATE_IMAGE_URL)
#             response.raise_for_status()
#             image_bytes = response.content
#     except httpx.HTTPError as e:
#         logger.error(f"Failed to fetch image from URL: {e}")
#         raise HTTPException(status_code=500, detail="Failed to fetch image from PLATE_IMAGE_URL")
    
#     filename = os.path.basename(PLATE_IMAGE_URL)
    
#     processed_results = await process_image_detection(
#         image_bytes=image_bytes,
#         filename=filename,
#         camera_id=camera_id
#     )
    
#     return {"processed_plates": processed_results}


# @router.post("/recognize_plate_externally/{plate_id}")
# async def recognize_plate_externally(plate_id: int = Path(..., title="The ID of the plate to recognize externally")):
#     """
#     Send an existing plate record to external recognition API.
#     """
#     async with async_session() as session:
#         result = await session.execute(select(Plate).where(Plate.id == plate_id))
#         plate = result.scalars().first()
        
#         if not plate:
#             raise HTTPException(status_code=404, detail=f"Plate with id {plate_id} not found")
        
#         # Build payload from existing record
#         bbox = []
#         if plate.bbox:
#             try:
#                 bbox = [int(x) for x in plate.bbox.split(',')]
#             except (ValueError, AttributeError):
#                 bbox = []
        
#         created_at = plate.created_at or datetime.now(timezone(timedelta(hours=6)))
        
#         payload = build_detection_payload(
#             plate_id=plate.id,
#             filename=plate.filename or f"{plate_id}.jpg",
#             detected_text=plate.detected_text,
#             confidence=float(plate.confidence) if isinstance(plate.confidence, str) else plate.confidence,
#             created_at=created_at,
#             camera_id=str(plate.camera_id),
#             bbox=bbox,
#             image_bytes=plate.image
#         )
        
#         return await send_to_external_api(plate_id, payload)


# @router.get("/plates/{plate_id}")
# async def get_plate(plate_id: int):
#     """Get a specific plate by ID."""
#     async with async_session() as session:
#         result = await session.execute(select(Plate).where(Plate.id == plate_id))
#         plate = result.scalars().first()
        
#         if not plate:
#             raise HTTPException(status_code=404, detail=f"Plate with id {plate_id} not found")
        
#         # Build response
#         bbox = []
#         if plate.bbox:
#             try:
#                 bbox = [int(x) for x in plate.bbox.split(',')]
#             except (ValueError, AttributeError):
#                 bbox = []
        
#         created_at = plate.created_at or datetime.now(timezone(timedelta(hours=6)))
#         if created_at.tzinfo is None:
#             created_at = created_at.replace(tzinfo=timezone(timedelta(hours=6)))
#         created_at_iso = created_at.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        
#         return {
#             "id": plate.id,
#             "filename": plate.filename,
#             "detectedText": plate.detected_text,
#             "confidence": float(plate.confidence) if isinstance(plate.confidence, str) else plate.confidence,
#             "createdAt": created_at_iso,
#             "cameraId": plate.camera_id,
#             "bbox": plate.bbox or "",
#             "image": base64.b64encode(plate.image).decode('utf-8') if plate.image else ""
#         }

