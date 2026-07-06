import json
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from ultralytics import YOLO

from .schemas import BoundingBox, Detection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best.pt"
CLASS_MAP_PATH = PROJECT_ROOT / "config" / "class_map.json"

MODEL_NAME = "mine-yolo-v8"
MODEL_VERSION = "1.0.0"


def select_inference_device() -> str:
    """Prefer the first CUDA GPU and fall back to CPU only when unavailable."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_class_map(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(
            "Required class map not found: "
            f"{path}. Expected config/class_map.json."
        )

    with path.open("r", encoding="utf-8-sig") as file:
        raw_map = json.load(file)

    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("class_map.json must be a non-empty JSON object")

    class_map: dict[int, str] = {}
    for raw_class_id, raw_class_name in raw_map.items():
        try:
            class_id = int(raw_class_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid class id: {raw_class_id!r}") from error

        if class_id < 0:
            raise ValueError(f"Class id must be non-negative: {class_id}")
        if not isinstance(raw_class_name, str) or not raw_class_name.strip():
            raise ValueError(f"Invalid class name for class id {class_id}")
        if class_id in class_map:
            raise ValueError(f"Duplicate class id: {class_id}")

        class_map[class_id] = raw_class_name.strip()

    expected_ids = list(range(len(class_map)))
    if sorted(class_map) != expected_ids:
        raise ValueError(
            f"Class ids must be contiguous and start at 0; expected {expected_ids}"
        )

    return dict(sorted(class_map.items()))


class Detector:
    def __init__(self) -> None:
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                "Required model weights not found: "
                f"{WEIGHTS_PATH}. Expected weights/best.pt; "
                "automatic weight download is disabled."
            )

        self.model_name = MODEL_NAME
        self.model_version = MODEL_VERSION
        self.weights_path = WEIGHTS_PATH
        self.class_map = load_class_map(CLASS_MAP_PATH)
        self.device = select_inference_device()
        self._model = YOLO(str(self.weights_path), task="detect")
        self._model.to(self.device)
        self._inference_lock = threading.Lock()

        # Only the model's class count is checked. Class names always come from
        # config/class_map.json and never from the weight metadata.
        model_class_count = len(self._model.names)
        if model_class_count != len(self.class_map):
            raise ValueError(
                "Class count mismatch: "
                f"weights contain {model_class_count} classes, "
                f"class_map.json contains {len(self.class_map)}"
            )

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> tuple[list[Detection], float, list[str]]:
        width, height = image.size
        predict_options: dict[str, Any] = {
            "source": image,
            "conf": confidence_threshold,
            "iou": iou_threshold,
            "verbose": False,
        }
        predict_options["device"] = self.device

        # One model instance is shared by all requests. Serializing predict calls
        # avoids concurrent mutation inside Ultralytics model internals.
        with self._inference_lock:
            started_at = time.perf_counter()
            result = self._model.predict(**predict_options)[0]
            latency_ms = (time.perf_counter() - started_at) * 1000.0

        detections: list[Detection] = []
        warnings: list[str] = []
        boxes = result.boxes
        if boxes is None:
            return detections, round(latency_ms, 3), warnings

        xyxy_values = boxes.xyxy.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()

        for detection_id, (xyxy, raw_class_id, confidence) in enumerate(
            zip(xyxy_values, class_ids, confidences, strict=True),
            start=1,
        ):
            class_id = int(raw_class_id)
            class_name = self.class_map.get(class_id)
            if class_name is None:
                class_name = "unknown"
                warnings.append(
                    f"detection_id={detection_id}: class_id={class_id} is not defined "
                    "in config/class_map.json; class_name was set to 'unknown'."
                )

            x1 = min(max(float(xyxy[0]), 0.0), float(width))
            y1 = min(max(float(xyxy[1]), 0.0), float(height))
            x2 = min(max(float(xyxy[2]), 0.0), float(width))
            y2 = min(max(float(xyxy[3]), 0.0), float(height))

            detections.append(
                Detection(
                    detection_id=detection_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=round(float(confidence), 6),
                    bbox_xyxy=BoundingBox(
                        x1=round(x1, 4),
                        y1=round(y1, 4),
                        x2=round(x2, 4),
                        y2=round(y2, 4),
                    ),
                    bbox_normalized=BoundingBox(
                        x1=round(x1 / width, 6),
                        y1=round(y1 / height, 6),
                        x2=round(x2 / width, 6),
                        y2=round(y2 / height, 6),
                    ),
                    area_px=round(max(0.0, x2 - x1) * max(0.0, y2 - y1), 4),
                )
            )

        return detections, round(latency_ms, 3), warnings
