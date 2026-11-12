import csv
import json

import cv2
import numpy as np
import pytest

from olholimpo.inference import (
    DetectionResult, file_sha256, inference_defaults, process_video, save_result, validate_image,
)
from scripts.evaluate import match_counts, operating_metrics


def result(image):
    return DetectionResult(image.copy(), [{"class_id": 0, "label": "residuo", "confidence": .8,
                                          "bbox_xyxy": [1, 2, 10, 12]}],
                           {"total": 5.}, {"sha256": "teste"}, image.shape[1], image.shape[0], .25)


def test_export_preserves_boxes_and_original_image(tmp_path):
    image = np.zeros((20, 30, 3), np.uint8)
    files = save_result(result(image), tmp_path)
    data = json.loads(open(files["json"], encoding="utf-8").read())
    assert data["image_width"] == 30
    assert data["detections"][0]["bbox_xyxy"] == [1, 2, 10, 12]
    with open(files["csv"], encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["x1"] == "1" and row["y2"] == "12"
    assert cv2.imread(files["image"]).shape == image.shape


@pytest.mark.parametrize("image", [None, np.zeros((2, 2)), np.zeros((2, 2, 3), np.float32)])
def test_rejects_invalid_pixel_layout(image):
    with pytest.raises(ValueError):
        validate_image(image)


def test_one_ground_truth_cannot_match_two_predictions():
    predictions = [{"confidence": .9, "bbox_xyxy": [0, 0, 10, 10]},
                   {"confidence": .8, "bbox_xyxy": [0, 0, 10, 10]}]
    assert match_counts(predictions, [[0, 0, 10, 10]], .5) == (1, 1, 0)
    assert match_counts(predictions, [[0, 0, 10, 10]], .95) == (0, 0, 1)
    metrics = operating_metrics([{"predictions": predictions, "ground_truth": [[0, 0, 10, 10]]}], .5)
    assert metrics["precision"] == .5 and metrics["recall"] == 1.


def test_inference_settings_cannot_be_applied_to_other_weights(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    settings = weights.with_suffix(".json")
    settings.write_text(json.dumps({"sha256": file_sha256(weights), "imgsz": 960,
                                    "confidence": .4}), encoding="utf-8")
    assert inference_defaults(weights) == {"imgsz": 960, "confidence": .4}
    weights.write_bytes(b"other checkpoint")
    with pytest.raises(ValueError, match="outro checkpoint"):
        inference_defaults(weights)


def test_video_export_keeps_source_fps_and_all_frames(tmp_path):
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10., (32, 24))
    for _ in range(5):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()

    class TestDetector:
        def predict(self, image, confidence):
            return result(image)

    files = process_video(path, tmp_path / "output", TestDetector())
    assert files["summary"]["frames"] == 5
    assert files["summary"]["source_fps"] == 10
    capture = cv2.VideoCapture(files["video"])
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 5
    capture.release()
