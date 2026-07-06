import io
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

# Keep Pillow's original decoder before importing Ultralytics. Ultralytics
# replaces Image.open with a helper that may try to install optional packages.
_PIL_IMAGE_OPEN = Image.open

from .detector import Detector
from .schemas import DetectResponse, HealthResponse, MetadataResponse


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        app.state.detector = await run_in_threadpool(Detector)
    except (FileNotFoundError, ValueError) as error:
        logger.critical("Detector startup failed: %s", error)
        raise
    logger.info("Detector initialized on inference device %s", app.state.detector.device)
    yield


app = FastAPI(
    title="Mine Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


def get_detector(request: Request) -> Detector:
    detector = getattr(request.app.state, "detector", None)
    if not isinstance(detector, Detector):
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return detector


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with _PIL_IMAGE_OPEN(io.BytesIO(image_bytes)) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
            image.load()
            return image
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image",
        ) from error


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    detector = get_detector(request)
    return HealthResponse(status="ok", model_loaded=detector is not None)


@app.get("/metadata", response_model=MetadataResponse)
async def metadata(request: Request) -> MetadataResponse:
    detector = get_detector(request)
    return MetadataResponse(
        model_name=detector.model_name,
        model_version=detector.model_version,
        device=detector.device,
        task="object_detection",
        weights="weights/best.pt",
        class_count=len(detector.class_map),
        class_map=detector.class_map,
    )


@app.post("/v1/detect", response_model=DetectResponse)
async def detect(
    request: Request,
    file: UploadFile = File(..., description="A single image file"),
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
    iou: float = Query(default=0.45, ge=0.0, le=1.0),
) -> DetectResponse:
    detector = get_detector(request)
    filename = file.filename or "uploaded_image"

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    await file.close()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    image = await run_in_threadpool(decode_image, image_bytes)

    try:
        detections, latency_ms, warnings = await run_in_threadpool(
            detector.detect,
            image,
            conf,
            iou,
        )
    except Exception as error:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail="Detection failed") from error

    return DetectResponse(
        model_name=detector.model_name,
        model_version=detector.model_version,
        image_filename=filename,
        image_width=image.width,
        image_height=image.height,
        confidence_threshold=conf,
        iou_threshold=iou,
        latency_ms=latency_ms,
        warnings=warnings,
        detections=detections,
    )
