"""
Entry point for the FastAPI application.
This script starts the application using Uvicorn.
"""

import uvicorn

from app.main import app

uvicorn.run(app, host="0.0.0.0", port=8088)
