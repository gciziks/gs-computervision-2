from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.presentation import project_payload
from olholimpo.inference import inference_defaults

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
IMAGE_LIMIT = 20 * 1024 * 1024
VIDEO_LIMIT = 100 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
OUTPUT_FILES = {"deteccoes.png", "deteccoes.json", "deteccoes.csv", "deteccoes.mp4"}
SAMPLE_NAMES = {
    "embalagem": "Embalagem na vegetação",
    "multiplos": "Vários resíduos no solo",
    "saco": "Saco e papelão na calçada",
}


class AnalysisError(Exception):
    """Falha com mensagem pronta para a pessoa que usa a interface."""


def _failure(error: Exception) -> str:
    if isinstance(error, AnalysisError):
        return str(error)
    if isinstance(error, FileNotFoundError):
        return ("Modelo ou arquivo não encontrado. Confira o caminho dos pesos e envie o "
                "arquivo novamente.")
    if isinstance(error, (ValueError, OSError)):
        return f"Não foi possível analisar o arquivo. {error}"
    LOGGER.exception("Falha no processamento local")
    return "O processamento falhou. Confira o erro no terminal, os pesos e a memória disponível."


def _duration(timings: dict) -> float | None:
    for key in ("total", "total_ms", "end_to_end"):
        if key in timings:
            return round(float(timings[key]), 1)
    values = [float(value) for value in timings.values() if isinstance(value, (int, float))]
    return round(sum(values), 1) if values else None


def _browser_video(path: Path) -> bool:
    """Converte o MP4 do OpenCV para H.264, formato que os navegadores reproduzem."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    converted = path.with_name("web.mp4")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-an", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted)],
            check=True, capture_output=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        LOGGER.warning("Conversão H.264 indisponível para %s", path.name)
        converted.unlink(missing_ok=True)
        return False
    converted.replace(path)
    return True


class DemoController:
    def __init__(
        self,
        weights: str | Path = ROOT / "models" / "best.pt",
        device: str = "cpu",
        imgsz: int | None = None,
        output_dir: str | Path = ROOT / "outputs" / "demo",
    ):
        self.weights = Path(weights).resolve()
        self.device = device
        defaults = inference_defaults(self.weights)
        self.imgsz = imgsz or defaults["imgsz"]
        self.confidence = defaults["confidence"]
        self.output_dir = Path(output_dir).resolve()
        self._detector = None
        self._lock = Lock()

    def detector(self):
        with self._lock:
            if self._detector is None:
                from olholimpo.inference import Detector

                self._detector = Detector(
                    weights=str(self.weights), device=self.device, imgsz=self.imgsz
                )
            return self._detector

    def model_status(self) -> dict:
        base = {"weights": self.weights.name, "device": self.device,
                "confidence": self.confidence, "imgsz": self.imgsz}
        if self._detector is not None:
            info = self._detector.model_info
            return {**base, "state": "loaded",
                    "weights": Path(info.get("path", self.weights)).name,
                    "device": str(info.get("device", self.device)),
                    "sha256": str(info.get("sha256", ""))[:12] or None,
                    "message": "Modelo carregado"}
        if not self.weights.is_file():
            return {**base, "state": "missing", "sha256": None,
                    "message": f"Pesos ausentes: {self.weights.name}. Consulte o README para "
                               "obter o checkpoint customizado. Nenhum modelo genérico será usado."}
        return {**base, "state": "idle", "sha256": None,
                "message": "Modelo pronto, carrega na primeira análise"}

    def _url(self, path: str | Path) -> str:
        path = Path(path)
        return f"/outputs/{path.parent.name}/{path.name}"

    def analyze_image(self, image: str | Path | np.ndarray | None, confidence: float) -> dict:
        if image is None:
            raise AnalysisError("Selecione uma imagem antes de analisar.")
        try:
            if isinstance(image, (str, Path)):
                from olholimpo.inference import read_image

                pixels = read_image(image)
            else:
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError("Use uma imagem RGB de três canais.")
                if image.shape[0] * image.shape[1] > 40_000_000:
                    raise ValueError("A imagem ultrapassa 40 milhões de pixels. Reduza a resolução.")
                pixels = np.ascontiguousarray(image[:, :, ::-1])
            from olholimpo.inference import save_result

            result = self.detector().predict(pixels, confidence=float(confidence))
            files = save_result(result, self.output_dir)
        except Exception as error:
            raise AnalysisError(_failure(error)) from error
        detections = [
            {"label": "resíduo", "confidence": round(float(item["confidence"]), 4),
             "bbox": [round(float(value), 1) for value in item["bbox_xyxy"]]}
            for item in result.detections
        ]
        return {
            "kind": "image",
            "count": len(detections),
            "detections": detections,
            "confidence": float(confidence),
            "duration_ms": _duration(result.timings_ms),
            "width": int(pixels.shape[1]),
            "height": int(pixels.shape[0]),
            "annotated": np.ascontiguousarray(result.annotated[:, :, ::-1]),
            "files": {key: self._url(files[key]) for key in ("image", "json", "csv")},
            "model": self.model_status(),
        }

    def analyze_video(self, video: str | Path | None, confidence: float) -> dict:
        if not video:
            raise AnalysisError("Selecione um vídeo antes de analisar.")
        path = Path(video)
        if not path.is_file():
            raise AnalysisError("O arquivo de vídeo não está disponível. Envie-o novamente.")
        if path.stat().st_size > VIDEO_LIMIT:
            raise AnalysisError("O vídeo ultrapassa 100 MB. Envie um trecho menor.")
        try:
            from olholimpo.inference import process_video

            files = process_video(str(path), self.output_dir, self.detector(),
                                  confidence=float(confidence), max_seconds=30)
        except Exception as error:
            raise AnalysisError(_failure(error)) from error
        playable = _browser_video(Path(files["video"]))
        summary = files["summary"]
        return {
            "kind": "video",
            "confidence": float(confidence),
            "playable": playable,
            "summary": {key: summary.get(key) for key in (
                "frames", "source_fps", "duration_seconds", "processing_seconds",
                "processing_fps", "p50_ms", "p95_ms", "detections")},
            "files": {key: self._url(files[key]) for key in ("video", "json", "csv")},
            "model": self.model_status(),
        }


def _samples(root: Path) -> dict[str, Path]:
    folder = root / "data" / "demo"
    if not folder.is_dir():
        return {}
    return {path.name: path for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES | {".mp4"}}


def _sample_title(name: str) -> str:
    stem = Path(name).stem
    for prefix, title in SAMPLE_NAMES.items():
        if stem.startswith(prefix):
            return title
    if stem.startswith("video"):
        return "Montagem de imagens do teste"
    return stem.replace("-", " ").capitalize()


async def _store_upload(upload, destination: Path, limit: int, suffixes: set[str]) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in suffixes:
        raise AnalysisError("Formato não suportado. Use " + ", ".join(sorted(suffixes)) + ".")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{uuid4().hex}{suffix}"
    size = 0
    with target.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                break
            stream.write(chunk)
    if size > limit:
        target.unlink(missing_ok=True)
        raise AnalysisError(f"O arquivo ultrapassa {limit // (1024 * 1024)} MB. "
                            "Envie um arquivo menor.")
    return target


def create_app(controller: DemoController, root: Path = ROOT):
    app = FastAPI(title="OlhoLimpo", docs_url=None, redoc_url=None, openapi_url=None)
    uploads = controller.output_dir / "_uploads"
    figures = {
        "results": root / "reports/training/results.png",
        "pr": root / "reports/training/BoxPR_curve.png",
        "f1": root / "reports/training/BoxF1_curve.png",
        "confusion": root / "reports/training/confusion_matrix_normalized.png",
        "comparison": root / "reports/evaluation/comparacao-validacao.png",
        "falha-000115": root / "assets/presentation/falha-000115.jpg",
    }
    report = root / "output/pdf/olholimpo-relatorio.pdf"

    def problem(error: AnalysisError):
        return JSONResponse({"error": str(error), "model": controller.model_status()}, 422)

    def checked(confidence: float):
        if not 0.05 <= confidence <= 0.95:
            raise AnalysisError("A confiança mínima deve ficar entre 0,05 e 0,95.")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    def status():
        return controller.model_status()

    @app.get("/api/project")
    def project():
        payload = project_payload(root, {key for key, path in figures.items() if path.is_file()})
        if report.is_file():
            # O PDF gerado localmente abre mesmo sem internet durante a apresentação.
            payload["links"]["report"] = "/relatorio.pdf"
        return payload

    @app.get("/relatorio.pdf", include_in_schema=False)
    def report_pdf():
        if not report.is_file():
            raise HTTPException(404)
        return FileResponse(report, media_type="application/pdf",
                            content_disposition_type="inline", filename="olholimpo-relatorio.pdf")

    @app.get("/api/samples")
    def samples():
        return [{"name": name, "title": _sample_title(name),
                 "kind": "video" if path.suffix.lower() == ".mp4" else "image",
                 "url": f"/samples/{name}"} for name, path in _samples(root).items()]

    @app.get("/samples/{name}", include_in_schema=False)
    def sample(name: str):
        path = _samples(root).get(name)
        if path is None:
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/figures/{key}", include_in_schema=False)
    def figure(key: str):
        path = figures.get(key)
        if path is None or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/outputs/{run}/{name}", include_in_schema=False)
    def output(run: str, name: str):
        if not re.fullmatch(r"[0-9a-f]{32}", run) or name not in OUTPUT_FILES:
            raise HTTPException(404)
        path = controller.output_dir / run / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, filename=None if name.endswith(".mp4") else name)

    @app.post("/api/image")
    async def analyze_image(file: UploadFile = File(...), confidence: float = Form(...)):
        try:
            checked(confidence)
            path = await _store_upload(file, uploads, IMAGE_LIMIT, IMAGE_SUFFIXES)
            try:
                result = await run_in_threadpool(controller.analyze_image, path, confidence)
            finally:
                path.unlink(missing_ok=True)
        except AnalysisError as error:
            return problem(error)
        result.pop("annotated", None)
        return result

    @app.post("/api/video")
    async def analyze_video(file: UploadFile = File(...), confidence: float = Form(...)):
        try:
            checked(confidence)
            path = await _store_upload(file, uploads, VIDEO_LIMIT, VIDEO_SUFFIXES)
            try:
                return await run_in_threadpool(controller.analyze_video, path, confidence)
            finally:
                path.unlink(missing_ok=True)
        except AnalysisError as error:
            return problem(error)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def build_app(
    weights: str | Path = ROOT / "models" / "best.pt",
    device: str = "cpu",
    imgsz: int | None = None,
    output_dir: str | Path = ROOT / "outputs" / "demo",
):
    return create_app(DemoController(weights=weights, device=device, imgsz=imgsz,
                                     output_dir=output_dir))


def main():
    parser = argparse.ArgumentParser(description="Aplicação local do detector OlhoLimpo.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--device", default="cpu", help="cpu ou índice CUDA, por exemplo 0")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    import uvicorn

    app = build_app(args.weights, args.device, args.imgsz)
    print(f"OlhoLimpo disponível em http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
