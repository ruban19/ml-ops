from fastapi import APIRouter

from app.api.api_v1 import routes

api_router = APIRouter()

api_router.include_router(routes.router, tags=["Inference EndPoint"])
