import argparse
import json
from pathlib import Path

from olholimpo.inference import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Compara resolução na validação, sem abrir o teste.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    from ultralytics import YOLO

    summary = json.loads((ROOT / "reports/training.json").read_text("utf-8"))
    weights = ROOT / "models/best.pt"
    if file_sha256(weights) != summary["sha256"]:
        raise ValueError("O checkpoint difere do treinamento registrado.")
    baseline_size = summary["config"]["imgsz"]
    if args.imgsz == baseline_size:
        parser.error("Escolha uma resolução diferente da utilizada no treinamento.")
    metrics = YOLO(str(weights)).val(
        data=str(ROOT / "data/processed/taco/data.yaml"), split="val", imgsz=args.imgsz,
        batch=8, device=args.device, workers=0, conf=.001, project=str(ROOT / "runs/resolution"),
        name=f"val{args.imgsz}", plots=False, verbose=False,
    )
    baseline = summary["validation_metrics"]
    selected = args.imgsz if metrics.box.map > baseline["metrics/mAP50-95(B)"] + .005 else baseline_size
    result = {
        "selection_split": "val", "checkpoint_sha256": summary["sha256"],
        "candidates": [
            {"imgsz": baseline_size, "map50": baseline["metrics/mAP50(B)"],
             "map50_95": baseline["metrics/mAP50-95(B)"]},
            {"imgsz": args.imgsz, "map50": float(metrics.box.map50),
             "map50_95": float(metrics.box.map)},
        ],
        "selected_imgsz": selected,
        "rule": "Adotar a alternativa somente se AP50-95 melhorar mais de 0,005 na validação.",
    }
    destination = ROOT / "reports/resolution-selection.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
