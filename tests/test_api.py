import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app import main as main_module


class FakeDetector:
    def __init__(self) -> None:
        self.model_name = "test-model"
        self.model_version = "test-version"
        self.device = "cuda:0"
        self.class_map = {0: "m2a4"}

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> tuple[list, float, list[str]]:
        return [], 1.25, ["fake warning"]


class DetectApiAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector_patch = patch.object(main_module, "Detector", FakeDetector)
        self.detector_patch.start()
        self.client_context = TestClient(main_module.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.detector_patch.stop()

    def test_rejects_non_image_content_type(self) -> None:
        response = self.client.post(
            "/v1/detect",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "Uploaded file must be an image")

    def test_rejects_invalid_bytes_advertised_as_image(self) -> None:
        response = self.client.post(
            "/v1/detect",
            files={"file": ("broken.jpg", b"not a jpeg", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "Uploaded file is not a valid image",
        )

    def test_accepts_a_valid_image(self) -> None:
        image_buffer = io.BytesIO()
        Image.new("RGB", (32, 24), color=(0, 0, 0)).save(
            image_buffer,
            format="JPEG",
        )

        response = self.client.post(
            "/v1/detect?conf=0.3&iou=0.5",
            files={"file": ("valid.jpg", image_buffer.getvalue(), "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["image_filename"], "valid.jpg")
        self.assertEqual(body["image_width"], 32)
        self.assertEqual(body["image_height"], 24)
        self.assertEqual(body["warnings"], ["fake warning"])

    def test_metadata_reports_selected_device(self) -> None:
        response = self.client.get("/metadata")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["device"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
