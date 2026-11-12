from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import time
from olholimpo.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def portable_metadata(value):
    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_metadata(item) for item in value]
    if isinstance(value, Path):
        value = value.as_posix()
    if isinstance(value, str):
        if value in {str(ROOT), ROOT.as_posix()}:
            return "."
        for prefix in (str(ROOT) + "\\", str(ROOT) + "/", ROOT.as_posix() + "/"):
            if value.startswith(prefix):
                return value[len(prefix):].replace("\\", "/")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina o detector customizado e preserva sua configuração."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/training")
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("epochs", "batch", "imgsz", "workers", "device", "name"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("data", "project"):
        path = Path(config[key])
        if not path.is_absolute():
            config[key] = str(ROOT / path)
    model_name = config.pop("model")
    manifest_path = ROOT / "data/manifests/taco.json"
    dataset_summary = json.loads((ROOT / "data/manifests/summary.json").read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(manifest_path)
    annotations_sha256 = dataset_summary["annotations_sha256"]
    started = datetime.now(timezone.utc).isoformat()
    before = time.perf_counter()
    model = YOLO(model_name)
    results = model.train(**config)
    elapsed = time.perf_counter() - before
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError("Treino terminou sem gerar best.pt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "results.csv",
        "results.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copyfile(source, report_dir / name)

    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    for key in checkpoint.keys() - {"model", "ema"}:
        checkpoint[key] = portable_metadata(checkpoint[key])
    for key in ("model", "ema"):
        module = checkpoint.get(key)
        if module is None:
            continue
        for attribute in ("args", "yaml", "yaml_file", "pt_path"):
            if hasattr(module, attribute):
                setattr(module, attribute, portable_metadata(getattr(module, attribute)))
    torch.save(checkpoint, args.output)

    resolved = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    (report_dir / "args.yaml").write_text(
        yaml.safe_dump(portable_metadata(resolved), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    with (run_dir / "results.csv").open(encoding="utf-8", newline="") as stream:
        epochs_completed = sum(1 for _ in csv.DictReader(stream))
    summary = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(elapsed, 3),
        "base_model": model_name,
        "config": portable_metadata(config),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else None,
        "epochs_completed": epochs_completed,
        "manifest_sha256": manifest_sha256,
        "annotations_sha256": annotations_sha256,
        "dataset_config_sha256": sha256_file(Path(config["data"])),
        "checkpoint": args.output.relative_to(ROOT).as_posix()
        if args.output.is_relative_to(ROOT)
        else args.output.name,
        "sha256": sha256_file(args.output),
        "checkpoint_metadata_sanitized": True,
        "run_directory": run_dir.relative_to(ROOT).as_posix()
        if run_dir.is_relative_to(ROOT)
        else run_dir.name,
        "validation_metrics": {key: float(value) for key, value in results.results_dict.items()}
        if results
        else {},
        "note": "Checkpoint selecionado por validação; o teste reservado é executado separadamente.",
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "reports/training.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".sha256").write_text(
        summary["sha256"] + "  " + args.output.name + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
