import sys
from types import SimpleNamespace

import numpy as np

from app.main import DemoController


def test_empty_input_does_not_load_model(tmp_path):
    controller = DemoController(weights=tmp_path / "absent.pt")

    image_result = controller.analyze_image(None, 0.25)
    video_result = controller.analyze_video(None, 0.25)

    assert image_result[:5] == (None, [], None, None, None)
    assert video_result[:4] == (None, None, None, None)
    assert controller._detector is None
    assert "Pesos ausentes" in image_result[-1]


def test_image_color_conversion_and_export(tmp_path, monkeypatch):
    class Detector:
        model_info = {"path": "best.pt", "sha256": "a" * 64, "device": "cpu"}

        def predict(self, image, confidence):
            assert image[0, 0].tolist() == [30, 20, 10]
            assert confidence == 0.4
            return SimpleNamespace(
                annotated=image,
                detections=[
                    {"label": "resíduo", "confidence": 0.8, "bbox_xyxy": [0, 0, 1, 1]}
                ],
                timings_ms={"total": 12.5},
            )

    def save_result(result, output_dir):
        assert output_dir.is_dir()
        return {"image": output_dir / "image.png", "json": output_dir / "result.json", "csv": output_dir / "result.csv"}

    monkeypatch.setitem(sys.modules, "olholimpo.inference", SimpleNamespace(save_result=save_result))
    controller = DemoController(output_dir=tmp_path)
    controller._detector = Detector()
    result = controller.analyze_image(np.array([[[10, 20, 30]]], dtype=np.uint8), 0.4)

    assert result[0][0, 0].tolist() == [10, 20, 30]
    assert result[1] == [["resíduo", 0.8, 0.0, 0.0, 1.0, 1.0]]
    assert "12.5 ms" in result[-2]
    assert "aaaaaaaaaaaa" in result[-1]


def test_invalid_image_clears_previous_output(tmp_path):
    controller = DemoController(output_dir=tmp_path)

    result = controller.analyze_image(np.zeros((3, 3), dtype=np.uint8), 0.25)

    assert result[:5] == (None, [], None, None, None)
    assert "três canais" in result[-2]
    assert not list(tmp_path.iterdir())


def test_oversized_video_rejected_before_loading_model(tmp_path):
    path = tmp_path / "large.mp4"
    with path.open("wb") as stream:
        stream.truncate(100 * 1024 * 1024 + 1)
    controller = DemoController(output_dir=tmp_path / "exports")

    result = controller.analyze_video(str(path), 0.25)

    assert result[:4] == (None, None, None, None)
    assert "100 MB" in result[-2]["erro"]
    assert controller._detector is None


def test_oversized_image_rejected_before_loading_model(tmp_path):
    path = tmp_path / "large.png"
    with path.open("wb") as stream:
        stream.truncate(20 * 1024 * 1024 + 1)
    controller = DemoController(output_dir=tmp_path / "exports")

    result = controller.analyze_image(str(path), 0.25)

    assert result[:5] == (None, [], None, None, None)
    assert "20 MB" in result[-2]
    assert controller._detector is None
