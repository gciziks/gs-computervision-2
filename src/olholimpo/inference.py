from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

MAX_PIXELS = 40_000_000


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_defaults(weights: str | Path) -> dict:
    path = Path(weights)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        return {"imgsz": 640, "confidence": .25}
    metadata = json.loads(metadata_path.read_text("utf-8"))
    if path.is_file() and metadata.get("sha256") != file_sha256(path):
        raise ValueError("A configuração de inferência pertence a outro checkpoint.")
    imgsz, confidence = int(metadata["imgsz"]), float(metadata["confidence"])
    if not 160 <= imgsz <= 1920 or not 0 < confidence < 1:
        raise ValueError("A configuração de inferência tem parâmetros inválidos.")
    return {"imgsz": imgsz, "confidence": confidence}


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Use uma imagem com três canais de cor.")
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("A imagem está vazia ou tem formato de pixels inválido.")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ValueError("A imagem excede o limite de 40 milhões de pixels.")


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise ValueError("Arquivo de imagem não encontrado.")
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("A imagem deve ter até 20 MB.")
    # imdecode também aceita caminhos com acentos no Windows.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    validate_image(image)
    return image


@dataclass
class DetectionResult:
    annotated: np.ndarray = field(repr=False)
    detections: list[dict]
    timings_ms: dict
    model: dict
    width: int
    height: int
    confidence: float
    processed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "image_width": self.width,
            "image_height": self.height,
            "threshold": self.confidence,
            "detections": self.detections,
            "timings_ms": self.timings_ms,
            "processed_at": self.processed_at,
            "bbox_format": "xyxy_pixels_top_left",
        }


class Detector:
    def __init__(self, weights="models/best.pt", device="cpu", imgsz=640):
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(
                f"Pesos customizados não encontrados em {path}. Execute o treinamento "
                "ou obtenha o checkpoint descrito em models/README.md."
            )
        if not 160 <= imgsz <= 1920:
            raise ValueError("A resolução de inferência deve estar entre 160 e 1920.")
        digest = file_sha256(path)
        checksum_path = path.with_suffix(".sha256")
        if checksum_path.is_file():
            tokens = checksum_path.read_text("utf-8").split()
            if not tokens or tokens[0] != digest:
                raise ValueError("O SHA-256 dos pesos difere do arquivo de verificação.")
        from ultralytics import YOLO

        self._model = YOLO(str(path))
        names = self._model.names
        if len(names) != 1 or str(names[0]).lower() not in {"residuo", "resíduo", "litter"}:
            raise ValueError("O checkpoint deve ter uma única classe customizada: residuo.")
        self._info = {
            "version": "olholimpo-0.1.0",
            "sha256": digest,
            "path": path.name,
            "device": str(device),
            "imgsz": imgsz,
            "class_names": {0: "residuo"},
        }
        self.device = device
        self.imgsz = imgsz
        self._lock = threading.Lock()
        self._model.predict(
            np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
            device=device, imgsz=imgsz, verbose=False,
        )
        if str(device) == "cpu":
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))

    @property
    def model_info(self):
        return dict(self._info)

    def predict(self, image: np.ndarray, confidence=0.25) -> DetectionResult:
        validate_image(image)
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("A confiança deve ser maior que zero e menor que um.")
        started = time.perf_counter()
        with self._lock:
            prediction = self._model.predict(
                image, conf=confidence, imgsz=self.imgsz, device=self.device, verbose=False,
            )[0]
        boxes = prediction.boxes
        height, width = image.shape[:2]
        detections = []
        annotated = image.copy()
        if boxes is not None:
            coordinates = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            for box, score in zip(coordinates, scores):
                x1, y1, x2, y2 = map(float, box)
                box = [max(0., min(x1, width)), max(0., min(y1, height)),
                       max(0., min(x2, width)), max(0., min(y2, height))]
                detections.append({
                    "class_id": 0, "label": "residuo", "confidence": float(score),
                    "bbox_xyxy": [round(value, 2) for value in box],
                })
                p1, p2 = tuple(map(int, box[:2])), tuple(map(int, box[2:]))
                cv2.rectangle(annotated, p1, p2, (100, 210, 80), 2)
                cv2.putText(annotated, f"residuo {score:.0%}", (p1[0], max(20, p1[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (100, 210, 80), 2)
        timings = {key: round(float(value), 3) for key, value in prediction.speed.items()}
        timings["total"] = round((time.perf_counter() - started) * 1000, 3)
        return DetectionResult(annotated, detections, timings, self.model_info,
                               width, height, float(confidence))


def _new_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir) / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_csv(path: Path, rows: list[dict], *, video=False):
    fields = (["frame_index", "timestamp_seconds"] if video else []) + [
        "class_id", "label", "confidence", "x1", "y1", "x2", "y2"
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row.update(zip(["x1", "y1", "x2", "y2"], row.pop("bbox_xyxy")))
            writer.writerow(row)


def save_result(result: DetectionResult, output_dir="outputs/demo") -> dict:
    destination = _new_output_dir(output_dir)
    image_path, json_path, csv_path = [destination / f"deteccoes.{ext}"
                                      for ext in ("png", "json", "csv")]
    success, encoded = cv2.imencode(".png", result.annotated)
    if not success:
        raise ValueError("Não foi possível salvar a imagem anotada.")
    encoded.tofile(image_path)
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), "utf-8")
    _write_csv(csv_path, result.detections)
    return {"image": str(image_path), "json": str(json_path), "csv": str(csv_path)}


def process_video(input_path, output_dir, detector, confidence=.25, max_seconds=30,
                  progress=None) -> dict:
    path = Path(input_path)
    if not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Envie um vídeo local de até 100 MB.")
    capture = cv2.VideoCapture(str(path))
    writer = None
    destination = None
    try:
        if not capture.isOpened():
            raise ValueError("Não foi possível abrir o vídeo. Tente um arquivo MP4.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError("O vídeo não tem uma taxa de quadros válida.")
        if frame_count <= 0 or frame_count / fps > max_seconds + .1:
            raise ValueError(f"O vídeo deve ter duração de até {max_seconds} segundos.")
        destination = _new_output_dir(output_dir)
        video_path = destination / "deteccoes.mp4"
        frames, rows, timings = [], [], []
        started = time.perf_counter()
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= int(max_seconds * fps) + 1:
                raise ValueError("O vídeo excede a duração permitida.")
            result = detector.predict(frame, confidence)
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (result.width, result.height))
                if not writer.isOpened():
                    raise ValueError("O codec de saída MP4 não está disponível.")
            writer.write(result.annotated)
            timings.append(result.timings_ms["total"])
            item = result.to_dict()
            item.update(frame_index=index, timestamp_seconds=round(index / fps, 4))
            frames.append(item)
            rows.extend({**box, "frame_index": index, "timestamp_seconds": round(index / fps, 4)}
                        for box in result.detections)
            index += 1
            if progress is not None:
                progress(min(index / frame_count, 1), desc=f"Quadro {index}")
        if not index:
            raise ValueError("O vídeo não contém quadros decodificáveis.")
        elapsed = time.perf_counter() - started
        summary = {
            "frames": index, "source_fps": fps, "duration_seconds": index / fps,
            "processing_seconds": elapsed, "processing_fps": index / elapsed,
            "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)),
            "detections": len(rows), "unique_objects_counted": False,
        }
        json_path, csv_path = destination / "deteccoes.json", destination / "deteccoes.csv"
        json_path.write_text(json.dumps({"summary": summary, "frames": frames},
                                        ensure_ascii=False, indent=2), "utf-8")
        _write_csv(csv_path, rows, video=True)
        return {"video": str(video_path), "json": str(json_path), "csv": str(csv_path),
                "summary": summary}
    finally:
        capture.release()
        if writer is not None:
            writer.release()
