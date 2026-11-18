import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import AnalysisError, DemoController, create_app


class FakeDetector:
    model_info = {"path": "best.pt", "sha256": "a" * 64, "device": "cpu"}

    def __init__(self, expected_pixel=None):
        self.expected_pixel = expected_pixel

    def predict(self, image, confidence):
        if self.expected_pixel is not None:
            assert image[0, 0].tolist() == self.expected_pixel
        return SimpleNamespace(
            annotated=image,
            detections=[{"label": "resíduo", "confidence": 0.8, "bbox_xyxy": [0, 0, 1, 1]}],
            timings_ms={"total": 12.5},
        )


def fake_save(result, output_dir):
    destination = output_dir / ("b" * 32)
    destination.mkdir(parents=True, exist_ok=True)
    return {key: destination / f"deteccoes.{ext}"
            for key, ext in (("image", "png"), ("json", "json"), ("csv", "csv"))}


def test_empty_input_does_not_load_model(tmp_path):
    controller = DemoController(weights=tmp_path / "absent.pt")

    with pytest.raises(AnalysisError, match="Selecione uma imagem"):
        controller.analyze_image(None, 0.25)
    with pytest.raises(AnalysisError, match="Selecione um vídeo"):
        controller.analyze_video(None, 0.25)
    assert controller._detector is None
    status = controller.model_status()
    assert status["state"] == "missing"
    assert "Pesos ausentes" in status["message"]


def test_image_color_conversion_and_export(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "olholimpo.inference", SimpleNamespace(save_result=fake_save))
    controller = DemoController(output_dir=tmp_path)
    controller._detector = FakeDetector(expected_pixel=[30, 20, 10])

    result = controller.analyze_image(np.array([[[10, 20, 30]]], dtype=np.uint8), 0.4)

    assert result["annotated"][0, 0].tolist() == [10, 20, 30]
    assert result["detections"] == [{"label": "resíduo", "confidence": 0.8, "bbox": [0.0, 0.0, 1.0, 1.0]}]
    assert result["count"] == 1
    assert result["duration_ms"] == 12.5
    assert result["files"]["csv"] == f"/outputs/{'b' * 32}/deteccoes.csv"
    assert result["model"]["sha256"] == "aaaaaaaaaaaa"


def test_invalid_image_creates_no_output(tmp_path):
    controller = DemoController(output_dir=tmp_path)

    with pytest.raises(AnalysisError, match="três canais"):
        controller.analyze_image(np.zeros((3, 3), dtype=np.uint8), 0.25)
    assert not list(tmp_path.iterdir())


def test_oversized_video_rejected_before_loading_model(tmp_path):
    path = tmp_path / "large.mp4"
    with path.open("wb") as stream:
        stream.truncate(100 * 1024 * 1024 + 1)
    controller = DemoController(output_dir=tmp_path / "exports")

    with pytest.raises(AnalysisError, match="100 MB"):
        controller.analyze_video(str(path), 0.25)
    assert controller._detector is None


def test_oversized_image_rejected_before_loading_model(tmp_path):
    path = tmp_path / "large.png"
    with path.open("wb") as stream:
        stream.truncate(20 * 1024 * 1024 + 1)
    controller = DemoController(output_dir=tmp_path / "exports")

    with pytest.raises(AnalysisError, match="20 MB"):
        controller.analyze_image(str(path), 0.25)
    assert controller._detector is None


@pytest.fixture
def client(tmp_path):
    controller = DemoController(weights=tmp_path / "absent.pt", output_dir=tmp_path / "exports")
    return TestClient(create_app(controller, root=tmp_path)), controller


def test_page_and_status_are_served(client):
    http, _ = client
    page = http.get("/")
    assert page.status_code == 200
    assert "OlhoLimpo" in page.text
    assert http.get("/static/app.js").status_code == 200
    assert http.get("/api/status").json()["state"] == "missing"


def test_upload_rejects_unknown_format_and_bad_threshold(client):
    http, controller = client
    wrong = http.post("/api/image", files={"file": ("nota.txt", b"x", "text/plain")},
                      data={"confidence": "0.35"})
    assert wrong.status_code == 422
    assert "Formato não suportado" in wrong.json()["error"]
    threshold = http.post("/api/image", files={"file": ("foto.png", b"x", "image/png")},
                          data={"confidence": "0.99"})
    assert threshold.status_code == 422
    assert "0,05 e 0,95" in threshold.json()["error"]
    assert controller._detector is None


def test_missing_weights_are_reported_without_generic_model(client):
    http, controller = client
    response = http.post("/api/image", files={"file": ("foto.png", b"not an image", "image/png")},
                         data={"confidence": "0.35"})
    assert response.status_code == 422
    assert response.json()["model"]["state"] == "missing"
    assert not list((controller.output_dir / "_uploads").iterdir())


def test_outputs_only_serve_known_files(client):
    http, controller = client
    run = controller.output_dir / ("c" * 32)
    run.mkdir(parents=True)
    (run / "deteccoes.json").write_text("{}", encoding="utf-8")
    (run / "segredo.txt").write_text("x", encoding="utf-8")
    assert http.get(f"/outputs/{'c' * 32}/deteccoes.json").status_code == 200
    assert http.get(f"/outputs/{'c' * 32}/segredo.txt").status_code == 404
    assert http.get("/outputs/nao-hex/deteccoes.json").status_code == 404
    assert http.get("/figures/qualquer").status_code == 404
    assert http.get("/samples/..%2Fsegredo.txt").status_code == 404


def test_local_report_is_linked_when_generated(tmp_path):
    controller = DemoController(weights=tmp_path / "absent.pt", output_dir=tmp_path / "exports")
    http = TestClient(create_app(controller, root=tmp_path))
    assert http.get("/relatorio.pdf").status_code == 404
    assert "report" not in http.get("/api/project").json()["links"]
    report = tmp_path / "output/pdf/olholimpo-relatorio.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"%PDF-1.4")
    response = http.get("/relatorio.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert http.get("/api/project").json()["links"]["report"] == "/relatorio.pdf"
