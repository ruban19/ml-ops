import os
from typing import List

import mlflow
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from starlette.responses import RedirectResponse
from transformers import pipeline

from logger import log

logger = log.setup_logging()
load_dotenv()
# instance of FastAPI Router
router = APIRouter()
mlflow.set_registry_uri(os.getenv("MLFLOW_URI"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))


@router.get("/", include_in_schema=False)
def redirect_to_docs(request: Request) -> RedirectResponse:
    root_path = request.scope.get("root_path", "").rstrip("/")
    return RedirectResponse(f"{root_path}/docs")


@router.get("/welcome", status_code=200)
def welcome():
    response = {
        "status": "Successfully Completed!",
        "data": [{"Hello": "World!"}],
    }
    logger.info("Welcome API request done!")
    return response


@router.post("/predict_finetuned", status_code=200)
def predict(input_data: List[str]):
    experiment_id = mlflow.get_experiment_by_name(
        os.getenv("EXPERIMENT_NAME")
    ).experiment_id
    best_runs = mlflow.search_runs(
        experiment_id, "", order_by=["metrics.accuracy DESC"], max_results=1
    )
    model_checkpoint = best_runs["params.Model Repo Name"][0]
    model_version = best_runs["params.Model Version"][0]
    classifier = pipeline(
        "text-classification", model=model_checkpoint, revision=model_version
    )
    result = classifier(input_data)
    for i, label in enumerate(result):
        label["text"] = input_data[i]
    response = {
        "status": "Model Inference Completed!",
        "data": result,
    }
    logger.info("Predict API Request Done!")
    return response
