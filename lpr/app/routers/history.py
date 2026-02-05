# from fastapi import APIRouter
# import base64
# from sqlalchemy.future import select
# from lpr.app.models import Plate
# from lpr.app.database import async_session

# router = APIRouter()

# @router.get("/history")
# async def get_history():
#     async with async_session() as session:
#         result = await session.execute(select(Plate).order_by(Plate.created_at.desc()))
#         plates = result.scalars().all()

#         processed_plates = []
#         for p in plates:
#             created_at_iso = p.created_at.isoformat(timespec='milliseconds').replace('+00:00', 'Z') if p.created_at else ""
#             processed_plates.append({
#                 "id": int(p.id),
#                 "imageId": int(p.id),
#                 "filename": p.filename or "",
#                 "detectedText": p.detected_text or "",
#                 "confidence": float(p.confidence) if p.confidence else 0.0,
#                 "createdAt": created_at_iso,
#                 "cameraId": p.camera_id or "",
#                 "trackId": p.track_id or 0,
#                 "bbox": "",
#                 "image": base64.b64encode(p.image).decode('utf-8') if p.image else ""
#             })

#         return {"processed_plates": processed_plates}
