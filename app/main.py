"""
Main FastAPI application setup.
This includes middleware configuration and router inclusion.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api_v1.api import api_router as api_v1_router

# App initialization
app = FastAPI(
    title="ML-Ops",
    description="Inference API for Text-Classifier",
)

# Add localhost to origins to allow bypassing CORS
origins = ["http://localhost:9500"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_v1_router)
