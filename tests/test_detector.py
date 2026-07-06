import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from app import detector as detector_module
from app import main as main_module


class TensorLike:
    def __init__(self, values: list) -> None:
        self.values = values

    def detach(self) -> "TensorLike":
        return self

    def cpu(self) -> "TensorLike":
        return self

    def tolist(self) -> list:
        return self.values


class FakeModel:
    def __init__(self) -> None:
        self.last_predict_options: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.last_predict_options = kwargs
        boxes = SimpleNamespace(
            xyxy=TensorLike([[10.0, 5.0, 30.0, 25.0]]),
            cls=TensorLike([99.0]),
            conf=TensorLike([0.8]),
        )
        return [SimpleNamespace(boxes=boxes)]


class DetectorAcceptanceTests(unittest.TestCase):
    def test_prefers_cuda_when_available(self) -> None:
        with patch.object(detector_module.torch.cuda, "is_available", return_value=True):
            self.assertEqual(detector_module.select_inference_device(), "cuda:0")

    def test_falls_back_to_cpu_when_cuda_is_unavailable(self) -> None:
        with patch.object(
            detector_module.torch.cuda,
            "is_available",
            return_value=False,
        ):
            self.assertEqual(detector_module.select_inference_device(), "cpu")

    def test_startup_fails_clearly_when_weights_are_missing(self) -> None:
        missing_weights = (
            detector_module.PROJECT_ROOT / "weights" / "__missing_best_for_test__.pt"
        )

        with patch.object(detector_module, "WEIGHTS_PATH", missing_weights):
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Required model weights not found",
            ):
                with TestClient(main_module.app):
                    pass

    def test_startup_fails_clearly_when_class_map_is_missing(self) -> None:
        available_weights = MagicMock()
        available_weights.is_file.return_value = True
        missing_class_map = (
            detector_module.PROJECT_ROOT
            / "config"
            / "__missing_class_map_for_test__.json"
        )

        with (
            patch.object(detector_module, "WEIGHTS_PATH", available_weights),
            patch.object(detector_module, "CLASS_MAP_PATH", missing_class_map),
        ):
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Required class map not found",
            ):
                with TestClient(main_module.app):
                    pass

    def test_unknown_class_id_returns_unknown_and_warning(self) -> None:
        detector = detector_module.Detector.__new__(detector_module.Detector)
        detector.class_map = {0: "m2a4"}
        detector.device = "cuda:0"
        fake_model = FakeModel()
        detector._model = fake_model
        detector._inference_lock = threading.Lock()

        detections, latency_ms, warnings = detector.detect(
            Image.new("RGB", (100, 50)),
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        self.assertGreaterEqual(latency_ms, 0.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 99)
        self.assertEqual(detections[0].class_name, "unknown")
        self.assertEqual(len(warnings), 1)
        self.assertIn("detection_id=1", warnings[0])
        self.assertIn("class_id=99", warnings[0])
        self.assertEqual(fake_model.last_predict_options["device"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
