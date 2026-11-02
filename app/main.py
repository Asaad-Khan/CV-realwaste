"""REST API for waste-material classification."""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.inference import WastePredictor


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Limit expensive processing to one request at a time for this CPU prototype.
INFERENCE_SLOT = threading.BoundedSemaphore(1)


class ClassScore(BaseModel):
    label: str
    score: float


class PredictionResponse(BaseModel):
    predicted_class: str
    top_predictions: list[ClassScore]
    model_name: str
    model_version: str
    processing_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.environ.get("MODEL_PATH")
    if not checkpoint:
        raise RuntimeError("Set MODEL_PATH to a trained checkpoint.")

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    app.state.predictor = WastePredictor(Path(checkpoint))
    LOGGER.info(
        "Loaded model version %s",
        app.state.predictor.metadata["run_id"],
    )

    yield

    del app.state.predictor


app = FastAPI(
    title="RealWaste Classification API",
    description=(
        "Predicts one of nine waste-material categories. "
        "Scores are uncalibrated softmax outputs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    return {
        "status": "ready",
        "model_name": "small_cnn",
        "model_version": predictor.metadata["run_id"],
        "class_count": len(predictor.classes),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request, file: UploadFile = File(...)):
    if not INFERENCE_SLOT.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Model is busy. Please retry shortly.",
            headers={"Retry-After": "1"},
        )

    start = time.perf_counter()

    try:
        payload = file.file.read(MAX_FILE_BYTES + 1)

        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(payload) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image file must be at most 5 MiB.",
            )

        try:
            with Image.open(BytesIO(payload)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="Supported image formats: JPEG, PNG, WEBP.",
                    )

                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image must contain at most 12 million pixels.",
                    )

                if getattr(image, "is_animated", False):
                    raise HTTPException(
                        status_code=415,
                        detail="Please upload a still image.",
                    )

                image.load()
                result = request.app.state.predictor.predict_image(image)

        except Image.DecompressionBombError as exc:
            raise HTTPException(
                status_code=413, detail="Image dimensions are too large."
            ) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="File is not a readable image."
            ) from exc

        result["processing_ms"] = round(
            (time.perf_counter() - start) * 1000, 2
        )
        LOGGER.info(
            "Prediction completed in %.2f ms",
            result["processing_ms"],
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        ) from exc
    finally:
        INFERENCE_SLOT.release()