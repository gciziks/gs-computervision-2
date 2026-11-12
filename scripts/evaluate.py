from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from olholimpo.inference import Detector, file_sha256, read_image, save_result

ROOT = Path(__file__).resolve().parents[1]


def image_paths(config: dict, split: str, yaml_path: Path) -> list[Path]:
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = yaml_path.parent / root
    entry = Path(config[split])
    path = entry if entry.is_absolute() else root / entry
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    result = []
    for line in path.read_text("utf-8").splitlines():
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        result.append(candidate.resolve())
    return result


def ground_truth(path: Path, shape: tuple) -> list[list[float]]:
    parts = list(path.parts)
    parts[parts.index("images")] = "labels"
    label = Path(*parts).with_suffix(".txt")
    if not label.is_file():
        raise ValueError(f"Anotação ausente: {label.name}")
    height, width = shape[:2]
    boxes = []
    for line in label.read_text("utf-8").splitlines():
        class_id, cx, cy, bw, bh = map(float, line.split())
        if class_id != 0:
            raise ValueError("A avaliação espera apenas a classe residuo.")
        boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height])
    return boxes


def overlap(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def match_counts(predictions, boxes, threshold, iou=.5):
    remaining = set(range(len(boxes)))
    true_positive = false_positive = 0
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        if prediction["confidence"] < threshold:
            continue
        best = max(remaining, key=lambda index: overlap(prediction["bbox_xyxy"], boxes[index]),
                   default=None)
        if best is not None and overlap(prediction["bbox_xyxy"], boxes[best]) >= iou:
            true_positive += 1
            remaining.remove(best)
        else:
            false_positive += 1
    return true_positive, false_positive, len(remaining)


def operating_metrics(records, threshold):
    tp, fp, fn = np.sum([match_counts(row["predictions"], row["ground_truth"], threshold)
                         for row in records], axis=0).tolist()
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn}


def predict_split(detector, paths):
    records = []
    for index, path in enumerate(paths):
        image = read_image(path)
        result = detector.predict(image, .01)
        records.append({"image": path.name, "predictions": result.detections,
                        "ground_truth": ground_truth(path, image.shape)})
        if (index + 1) % 25 == 0:
            print(f"Avaliadas {index + 1}/{len(paths)} imagens", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description="Seleciona limiar na validação e avalia teste reservado.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/taco/data.yaml")
    parser.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    training_path = ROOT / "reports/training.json"
    if not training_path.is_file():
        raise ValueError("Execute o treinamento antes da avaliação.")
    training_record = json.loads(training_path.read_text("utf-8"))
    if training_record.get("sha256") != file_sha256(args.weights):
        raise ValueError("Os pesos não correspondem ao treinamento registrado.")
    manifest_path = ROOT / "data/manifests/taco.json"
    expected_manifest = training_record.get("manifest_sha256")
    if expected_manifest and expected_manifest != file_sha256(manifest_path):
        raise ValueError("O manifesto mudou desde o treino. Não avalie com uma divisão diferente.")
    destination = ROOT / "reports/evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.data.read_text("utf-8"))
    paths = {split: image_paths(config, split, args.data) for split in ("train", "val", "test")}
    manifest = json.loads(manifest_path.read_text("utf-8"))
    data_root = Path(config["path"])
    for split in paths:
        expected_paths = {str((data_root / row["prepared_file"]).resolve())
                          for row in manifest if row["split"] == split}
        if set(map(str, paths[split])) != expected_paths:
            raise ValueError(f"A lista {split} não corresponde ao manifesto congelado.")
    for row in manifest:
        image_path = data_root / row["prepared_file"]
        parts = list(image_path.parts)
        parts[parts.index("images")] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        if file_sha256(image_path) != row["sha256"]:
            raise ValueError(f"A imagem {image_path.name} foi modificada depois do preparo.")
        if row.get("label_sha256") and file_sha256(label_path) != row["label_sha256"]:
            raise ValueError(f"A anotação de {image_path.name} foi modificada depois do preparo.")
    detector = Detector(args.weights, args.device, args.imgsz)
    validation = predict_split(detector, paths["val"])
    sweep = [{"confidence": round(value, 2), **operating_metrics(validation, value)}
             for value in np.arange(.05, .76, .05)]
    selected = max(sweep, key=lambda item: (item["f1"], item["precision"]))
    threshold = selected["confidence"]
    selection = {"split": "val", "iou": .5, "selected": selected, "sweep": sweep,
                 "checkpoint_sha256": detector.model_info["sha256"]}
    # Gravar a escolha antes de abrir o teste mantém explícita a ordem do protocolo.
    (destination / "threshold.json").write_text(json.dumps(selection, indent=2), "utf-8")
    test = predict_split(detector, paths["test"])
    operating = operating_metrics(test, threshold)
    from ultralytics import YOLO

    ap = YOLO(args.weights).val(data=str(args.data), split="test", device=args.device,
                               imgsz=args.imgsz, batch=8, workers=0, conf=.001,
                               project=str(ROOT / "runs/evaluation"), name="test",
                               plots=True, verbose=False)
    timings = []
    for path in paths["test"][:30]:
        image = read_image(path)
        before = time.perf_counter()
        detector.predict(image, threshold)
        timings.append((time.perf_counter() - before) * 1000)
    errors, figures = [], []
    for row, path in zip(test, paths["test"]):
        tp, fp, fn = match_counts(row["predictions"], row["ground_truth"], threshold)
        if (fp or fn) and len(errors) < 5:
            errors.append({"image": path.name, "true_positives": tp,
                           "false_positives": fp, "false_negatives": fn,
                           "description": f"Falsos positivos: {fp}; falsos negativos: {fn}. "
                                          "Limiar escolhido pela validação e IoU 0,50."})
    analysis_path = ROOT / "docs/ERROR-ANALYSIS.json"
    if analysis_path.is_file():
        notes = {item["image"]: item for item in json.loads(analysis_path.read_text("utf-8"))}
        for error in errors:
            if error["image"] in notes:
                error.update(notes[error["image"]])
    chosen = list(dict.fromkeys([row["image"] for row in errors] +
                               [row["image"] for row in test[:3]]))[:8]
    evidence_config = ROOT / "configs/evidence.json"
    if evidence_config.is_file():
        excluded = json.loads(evidence_config.read_text("utf-8"))["exclude_images"]
        chosen = [name for name in chosen if name not in excluded]
    for path in paths["test"]:
        if path.name not in chosen:
            continue
        image = read_image(path)
        result = detector.predict(image, threshold)
        for box in ground_truth(path, image.shape):
            cv2.rectangle(result.annotated, tuple(map(int, box[:2])), tuple(map(int, box[2:])),
                          (220, 100, 30), 2)
        save_result(result, ROOT / "outputs/evaluation")
        figure_path = ROOT / "outputs/evaluation" / f"exemplo-{path.stem}.png"
        cv2.imencode(".png", result.annotated)[1].tofile(figure_path)
        figures.append({"path": figure_path.relative_to(ROOT).as_posix(),
                        "caption": f"{path.name}: verde = predição; azul = anotação. "
                                   "Imagem do teste interno TACO."})
    train_config = training_record.get("config", {})
    history_path = ROOT / "reports/training/results.csv"
    import csv

    with history_path.open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    training = {
        "base_model": training_record.get("base_model"),
        "epochs_requested": train_config.get("epochs"), "epochs_completed": len(history),
        "batch": train_config.get("batch"), "imgsz": train_config.get("imgsz"),
        "seed": train_config.get("seed"), "duration_seconds": training_record.get("duration_seconds"),
        "gpu": training_record.get("gpu"), "torch": training_record.get("torch"),
        "ultralytics": training_record.get("ultralytics"),
        "results_csv": "reports/training/results.csv",
    }
    summary_path = ROOT / "data/manifests/summary.json"
    data_summary = json.loads(summary_path.read_text("utf-8")) if summary_path.exists() else {}
    counts = {}
    for split, items in paths.items():
        counts[f"{split}_images"] = len(items)
        counts[f"{split}_instances"] = sum(len(ground_truth(path, read_image(path).shape))
                                           for path in items)
    output = {
        "model": {"name": "YOLO26n — residuo", "sha256": detector.model_info["sha256"],
                  "weights": "models/best.pt"},
        "dataset": {**counts, "annotation_sha256": data_summary.get("annotations_sha256"),
                    "source": "TACO", "independent_negatives": 0,
                    "note": "Sem ensaio local independente ou coleta própria nesta execução."},
        "evaluation": {**operating, "map50": float(ap.box.map50), "map50_95": float(ap.box.map),
                       "confidence": threshold, "iou": .5,
                       "selection_split": "val", "evaluation_split": "test",
                       "ap_note": "AP integrada pela Ultralytics; precisão/recall/F1 em limiar fixo, "
                                  "com pareamento por confiança e IoU 0,50."},
        "benchmark": {"device": str(args.device), "imgsz": args.imgsz, "images": len(timings),
                      "warmup": "modelo aquecido antes da medição",
                      "p50_ms": float(np.percentile(timings, 50)),
                      "p95_ms": float(np.percentile(timings, 95)),
                      "fps": 1000 / float(np.mean(timings)),
                      "scope": "Inferência + anotação em imagens pré-decodificadas; sem captura/exibição."},
        "training": training, "figures": figures, "errors": errors,
        "limitations": ["Não houve teste com negativos independentes.",
                        "Licenças de imagem têm metadados incompletos; ver data card.",
                        "Webcam e cenário urbano real ainda precisam de ensaio físico."]}
    if min(operating["precision"], operating["recall"], float(ap.box.map50)) < .75:
        output["limitations"].append(
            "Precisão, recall e AP50 ficaram abaixo das metas internas de 0,75; "
            "não há alegação de acurácia operacional."
        )
    for curve in ("results.png", "BoxPR_curve.png", "BoxF1_curve.png"):
        curve_path = ROOT / "reports/training" / curve
        if curve_path.is_file():
            figures.append({"path": curve_path.relative_to(ROOT).as_posix(),
                            "caption": f"Treinamento e validação: {curve}. "
                                       "Não são métricas do teste reservado."})
    baseline_path = ROOT / "reports/baseline/summary.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text("utf-8"))
        if baseline.get("manifest_sha256") == training_record.get("manifest_sha256"):
            key = "metrics/mAP50(B)"
            experiments = [
                {"name": "Referência inicial", "epochs": baseline["epochs_completed"],
                 "batch": baseline["config"]["batch"],
                 "validation_map50": baseline["validation_metrics"][key]},
                {"name": "Ajuste completo", "epochs": len(history),
                 "batch": train_config["batch"],
                 "validation_map50": training_record["validation_metrics"][key]},
            ]
            output["experiments"] = experiments
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(8, 4))
            bars = axis.bar([item["name"] for item in experiments],
                            [item["validation_map50"] for item in experiments],
                            color=["#8eaca0", "#21604b"])
            axis.set(ylim=(0, 1), ylabel="AP50 na validação", title="Evolução do ajuste customizado")
            axis.bar_label(bars, fmt="%.3f", padding=4)
            figure.tight_layout()
            comparison_path = destination / "comparacao-validacao.png"
            figure.savefig(comparison_path, dpi=180)
            plt.close(figure)
            figures.append({"path": comparison_path.relative_to(ROOT).as_posix(),
                            "caption": "Comparação na mesma validação, sem usar o teste. "
                                       "Referência com uma época e batch 8; ajuste completo com "
                                       "batch 16. Não é uma ablação isolada do número de épocas."})
    (ROOT / "reports/metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), "utf-8")
    Path(args.weights).with_suffix(".json").write_text(json.dumps({
        "sha256": detector.model_info["sha256"], "imgsz": args.imgsz,
        "confidence": threshold, "class_names": {0: "residuo"},
        "selection_split": "val",
    }, ensure_ascii=False, indent=2), "utf-8")
    (destination / "test_predictions.json").write_text(json.dumps(test, indent=2), "utf-8")
    print(json.dumps(output["evaluation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
