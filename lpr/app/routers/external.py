import os
import cv2
import numpy as np
import base64
import httpx
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, UploadFile, File, HTTPException
from sqlalchemy.future import select
from lpr.app.ml_utils import ml_service
from lpr.app.database import async_session
from lpr.app.models import Plate
from uuid import uuid4

logger = logging.getLogger(__name__)

router = APIRouter()

EXTERNAL_API = os.getenv("EXTERNAL_API")
PLATE_IMAGE_URLL = os.getenv("PLATE_IMAGE_URLL")
PLATE_RECOGNITION_ADD_URL = os.getenv("PLATE_RECOGNITION_ADD_URL")
EXTERNAL_RECOGNITION_URL = os.getenv("EXTERNAL_RECOGNITION_URL")
CAMERA_LIST_URL = os.getenv("CAMERA_LIST_URL")

async def send_plate_result_to_external_recognition(client, plate):
    payload = {
        "id": plate["id"],
        "imageId": plate.get("imageId", plate["id"]),
        "detectedText": plate["detectedText"],
        "confidence": float(plate["confidence"]),
        "createdAt": plate["createdAt"],
        "bbox": plate.get("bbox", ""),
        "image": plate["image"],
        "overwrite": True
    }
    response = await client.post(EXTERNAL_RECOGNITION_URL, json=payload)
    return {"status": response.status_code, "body": response.text}

async def send_plate_result_to_external_db(client, plate):
    payload = {
        "id": plate["id"],
        "filename": plate["filename"],
        "detectedText": plate["detectedText"],
        "confidence": float(plate["confidence"]),
        "createdAt": plate["createdAt"],
        "cameraId": 1 if plate["cameraId"] == "external" else 0,
        "bbox": plate.get("bbox", ""),
        "image": plate["image"],
        "overwrite": True
    }
    response = await client.post(PLATE_RECOGNITION_ADD_URL, json=payload)
    return {"status": response.status_code, "body": response.text}

async def recognize_and_store_plate(image_bytes, filename, camera_id="local", external_image_id=None, send_highest_confidence_only=False):
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if image is None:
        return []
    temp_path = f"tmp_{uuid4()}.jpg"
    cv2.imwrite(temp_path, image)
    detections = ml_service.detect_plates(temp_path)
    os.remove(temp_path)
    results = []
    async with async_session() as session:
        for plate in detections:
            cropped = plate["cropped_image"]
            text, (_, confs) = ml_service.recognize("lpr", cropped)
            confidence = float(np.mean([float(c) for c in confs])) if confs else 0.0
            last = await session.execute(select(Plate.track_id).order_by(Plate.track_id.desc()))
            track_id = (last.scalars().first() or 0) + 1
            crop_bytes = cv2.imencode(".jpg", cropped)[1].tobytes()
            created_at = datetime.now(timezone(timedelta(hours=6)))
            record = Plate(
                filename=filename,
                detected_text=text,
                confidence=str(confidence),
                camera_id=camera_id,
                image=crop_bytes,
                track_id=track_id,
                created_at=created_at,
                image_id=external_image_id  
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            created_iso = record.created_at.isoformat(timespec="milliseconds").replace("+00:00", "Z")
            results.append({
                "id": record.id,
                "imageId": external_image_id if external_image_id else record.id,
                "filename": filename,
                "detectedText": text,
                "confidence": confidence,
                "createdAt": created_iso,
                "cameraId": camera_id,
                "trackId": track_id,
                "bbox": "",
                "image": base64.b64encode(crop_bytes).decode()
            })
    
    if send_highest_confidence_only and results:
        highest_conf_plate = max(results, key=lambda x: x["confidence"])
        logger.info(f"[FILTER] Returning only highest confidence plate: '{highest_conf_plate['detectedText']}' ({highest_conf_plate['confidence']:.4f})")
        return [highest_conf_plate]
    
    return results

async def download_image(client, url, filename):

    if url.startswith('data:image'):
        try:
            base64_data = url.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            logger.info(f"[DOWNLOAD] Successfully decoded base64 data URL ({len(image_bytes)} bytes)")
            return {"content": image_bytes, "filename": filename}
        except Exception as e:
            logger.error(f"[DOWNLOAD ERROR] Failed to decode base64 data URL: {e}")
            return None
    
    try:
        response = await client.get(
            url,
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "image/*,*/*",
            }
        )
        response.raise_for_status()
        return {"content": response.content, "filename": filename}
    except httpx.TimeoutException as e:
        logger.error(f"[DOWNLOAD ERROR] Timeout while downloading {url}: {e}")
        return None
    except httpx.ConnectError as e:
        logger.error(f"[DOWNLOAD ERROR] Connection error while downloading {url}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"[DOWNLOAD ERROR] HTTP error {e.response.status_code} while downloading {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"[DOWNLOAD ERROR] Unexpected error while downloading {url}: {e}")
        return None

@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    local_results = await recognize_and_store_plate(image_bytes, file.filename, send_highest_confidence_only=True)
    async with httpx.AsyncClient(verify=False) as client:
        external_posts = []
        for plate in local_results:
            result = await send_plate_result_to_external_recognition(client, plate)
            external_posts.append(result)
    return {"local_results": local_results, "external_posts": external_posts}

async def process_remote_images_logic():

    results = []
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.get(EXTERNAL_API)
        data = response.json()
        images = [
            {
                "url": f"{PLATE_IMAGE_URLL.rstrip('/')}/{item['image'].lstrip('/')}" if PLATE_IMAGE_URLL else item["image"],
                "filename": item["image"].split("/")[-1],
                "external_id": item["id"]
            }
            for item in data.get("res", [])
        ]
        for img in images:
            downloaded = await download_image(client, img["url"], img["filename"])
            if not downloaded:
                continue
            plates = await recognize_and_store_plate(
                downloaded["content"],
                downloaded["filename"],
                camera_id="external",
                external_image_id=img["external_id"],
                send_highest_confidence_only=True 
            )
            posts = []
            for plate in plates:
                post_result = await send_plate_result_to_external_recognition(client, plate)
                posts.append(post_result)
            results.append({
                "filename": img["filename"],
                "external_id": img["external_id"],
                "plates": plates,
                "external_posts": posts
            })
    return {"total": len(results), "processed": results}

@router.post("/external/process_remote_images")
async def process_remote_images():

    return await process_remote_images_logic()

@router.post("/detect/camera")
async def detect_camera(camera_id: str):
    try:
        async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
            response = await client.get(CAMERA_LIST_URL)
            response.raise_for_status()
            cameras = response.json()
            cam = next((c for c in cameras if str(c.get("id")) == str(camera_id)), None)
            if not cam:
                raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
            
            user = cam.get("rtspUser")
            password = cam.get("rtspPass")
            host = cam.get("rtspHost")
            port = cam.get("rtspPort", "554")
            path = cam.get("rtspPath", "/")
            
            rtsp_url = f"rtsp://{user}:{password}@{host}:{port}{path}"
            if path.lower().endswith('/streaming/channels'):
                rtsp_url = f"{rtsp_url}/101"
            else:
                rtsp_url = f"{rtsp_url}?channel=1&subtype=0"
        
        logger.info(f"[detect_camera] Connecting to RTSP: {rtsp_url[:50]}...")
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            logger.error(f"[detect_camera] Cannot open RTSP stream for camera {camera_id}")
            return {"error": f"Cannot open camera {camera_id}", "rtsp_url": rtsp_url}
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"[detect_camera] Cannot read frame from camera {camera_id}")
            return {"error": f"Cannot read frame from camera {camera_id}"}
        
        logger.info(f"[detect_camera] Frame captured successfully for camera {camera_id}, shape: {frame.shape}")
        
        _, buf = cv2.imencode(".jpg", frame)
        image_bytes = buf.tobytes()
        frame_base64 = base64.b64encode(image_bytes).decode()
        
        filename = f"camera_{camera_id}_{int(datetime.now().timestamp())}.jpg"
        
        plates = await recognize_and_store_plate(image_bytes=image_bytes, filename=filename, camera_id=camera_id)
        logger.info(f"[detect_camera] Detection complete, found {len(plates)} plates")
        
        return {"processed_plates": plates, "frame": frame_base64}
    except Exception as e:
        logger.error(f"[detect_camera] Error: {e}")
        return {"error": str(e)}

@router.post("/detect/image")
async def detect_image(request: dict):

    try:
        image_base64 = request.get("image_base64")
        if not image_base64:
            return {"error": "Missing 'image_base64' in request body"}
        
        image_bytes = base64.b64decode(image_base64)
        
        filename = f"upload_{int(datetime.now().timestamp())}.jpg"
        plates = await recognize_and_store_plate(
            image_bytes=image_bytes, 
            filename=filename, 
            camera_id="local",
            send_highest_confidence_only=True
        )
        return {"processed_plates": plates}
    except Exception as e:
        logger.error(f"detect_image error: {e}")
        return {"error": str(e)}

# @router.post("/external/webhook/new_image")
# async def webhook_new_image(image_data: dict):
#     logger.info(f"[WEBHOOK_NEW_IMAGE] Received payload: {image_data}")
#     external_id = image_data.get("id")
#     image_url = image_data.get("image")

#     if not image_url:
#         logger.error("[WEBHOOK_NEW_IMAGE] Missing 'image' field in payload")
#         raise HTTPException(status_code=400, detail="Missing 'image' field")

#     original_image_value = image_url
#     if not image_url.startswith(("http://", "https://", "data:image")) and PLATE_IMAGE_URLL:
#         image_url = f"{PLATE_IMAGE_URLL.rstrip('/')}/{image_url.lstrip('/')}"
#         logger.info(f"[WEBHOOK_NEW_IMAGE] Built full image URL from base: {image_url} (original='{original_image_value}')")

#     try:
#         if external_id is not None:
#             external_image_id = int(external_id)
#             if external_image_id <= 0:
#                 logger.error(f"[WEBHOOK_NEW_IMAGE] Invalid 'id' value (non-positive): {external_id}")
#                 raise HTTPException(status_code=400, detail="Invalid 'id' field - must be a positive integer")
#         else:
#             external_image_id = None
#     except (ValueError, TypeError):
#         logger.error(f"[WEBHOOK_NEW_IMAGE] Invalid 'id' value (not int): {external_id}")
#         raise HTTPException(status_code=400, detail="Invalid 'id' field - must be an integer")

#     filename = image_url.split("/")[-1] if "/" in image_url else image_url
#     logger.info(f"[WEBHOOK_NEW_IMAGE] Resolved filename: {filename}")

#     async with httpx.AsyncClient(verify=False) as client:
#         logger.info(f"[WEBHOOK_NEW_IMAGE] Downloading image from {image_url}")
#         downloaded = await download_image(client, image_url, filename)

#         if not downloaded:
#             logger.error(f"[WEBHOOK_NEW_IMAGE] Failed to download image from {image_url}")
#             raise HTTPException(status_code=400, detail=f"Failed to download image from {image_url}")

#         logger.info(f"[WEBHOOK_NEW_IMAGE] Image downloaded: {downloaded['filename']}, bytes={len(downloaded['content'])}")

#         logger.info("[WEBHOOK_NEW_IMAGE] Starting plate recognition")
#         plates = await recognize_and_store_plate(
#             downloaded["content"],
#             downloaded["filename"],
#             camera_id="external",
#             external_image_id=external_image_id,
#             send_highest_confidence_only=True 
#         )
#         logger.info(f"[WEBHOOK_NEW_IMAGE] Plate recognition finished, plates_found={len(plates)}")

#         posts = []
#         for plate in plates:
#             logger.info(
#                 f"[WEBHOOK_NEW_IMAGE] Plate detected: id={plate.get('id')} "
#                 f"imageId={plate.get('imageId')} text={plate.get('detectedText')} "
#                 f"confidence={plate.get('confidence')}"
#             )
#             post_result = await send_plate_result_to_external_recognition(client, plate)
#             logger.info(
#                 f"[WEBHOOK_NEW_IMAGE] Sent to external recognition: "
#                 f"status={post_result.get('status')} body={post_result.get('body')}"
#             )
#             posts.append(post_result)

#     response = {
#         "status": "processed",
#         "image_id": external_id,
#         "filename": filename,
#         "plates_found": len(plates),
#         "results": plates,
#         "external_posts": posts
#     }
#     logger.info(f"[WEBHOOK_NEW_IMAGE] Response: {response}")
#     return response