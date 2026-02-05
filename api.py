import os
import sys
import uvicorn
import warnings
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

warnings.filterwarnings("ignore", message="Importing from timm.models.helpers is deprecated", category=FutureWarning)

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), 'lpr'))

from lpr.app.models import Base
from lpr.app.database import engine, create_tables
from lpr.app.config import settings
from lpr.app.routers import detect, cameras, external

API_HOST = os.getenv("API_HOST")
API_PORT = int(os.getenv("API_PORT"))

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="License Plate Detection & Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "License Plate Recognition API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            # "detect": "/detect",
            "cameras": "/cameras",
            "external": "/external"
        }
    }

# app.include_router(detect.router)
app.include_router(cameras.router)
app.include_router(external.router)

@app.on_event("startup")
async def startup_event():
    await create_tables()
    logging.info("FastAPI server started")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
