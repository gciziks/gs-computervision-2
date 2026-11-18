from __future__ import annotations

import json
import math
from pathlib import Path


def load_records(root: Path) -> dict:
    records = {}
    issues = []
    for name, relative, expected in (
        ("team", "deliverables/team.json", list),
        ("metrics", "reports/metrics.json", dict),
        ("links", "deliverables/links.json", dict),
        ("resolution", "reports/resolution-selection.json", dict),
    ):
        try:
            value = json.loads((root / relative).read_text(encoding="utf-8"))
            if not isinstance(value, expected):
                raise ValueError("Formato inesperado")
            records[name] = value
        except (OSError, ValueError):
            records[name] = expected()
            issues.append(relative)
    records["issues"] = issues
    return records


def section(record: dict, key: str) -> dict:
    value = record.get(key, {})
    return value if isinstance(value, dict) else {}


def number(value, decimals=0, percent=False) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "Não registrado"
    rendered = f"{value * 100 if percent else value:,.{decimals}f}"
    return rendered.translate(str.maketrans({",": ".", ".": ","})) + ("%" if percent else "")


def error_figure(image, root: Path) -> str | None:
    # As figuras revisadas pertencem aos artefatos da apresentação, não a um caminho recebido da interface.
    approved = {"000115.jpg": root / "assets/presentation/falha-000115.jpg"}
    figure = approved.get(image)
    return str(figure) if figure is not None and figure.is_file() else None


def error_details(metrics: dict, image: str, root: Path) -> tuple[str, str | None]:
    errors = metrics.get("errors", [])
    case = next((row for row in errors if isinstance(row, dict) and row.get("image") == image), None)
    if case is None:
        return "Nenhum caso de erro registrado para esta imagem.", None
    text = (
        f"### Imagem {image}\n\n"
        f"**Acertos:** {number(case.get('true_positives'))} · "
        f"**Falsos positivos:** {number(case.get('false_positives'))} · "
        f"**Objetos não recuperados:** {number(case.get('false_negatives'))}\n\n"
        f"**Observação:** {case.get('observation', case.get('description', 'Não registrada.'))}\n\n"
        f"**Hipótese:** {case.get('hypothesis', 'Não registrada.')}\n\n"
        f"**Correção proposta:** {case.get('proposed_correction', 'Não registrada.')}"
    )
    figure = error_figure(image, root)
    if figure is not None:
        return text, str(figure)
    return text + "\n\nA fotografia deste caso não foi incluída na apresentação. Os resultados quantitativos permanecem registrados.", None




TARGET = 0.75
FIGURE_FOR_ERROR = {"000115.jpg": "falha-000115"}


def _text(value, fallback="Não registrado") -> str:
    return str(value) if isinstance(value, str) and value.strip() else fallback


def project_payload(root: Path, figures: set[str] | None = None) -> dict:
    """Reúne os registros da execução para a página do projeto, sem recalcular métricas."""
    figures = figures or set()
    records = load_records(root)
    metrics = records["metrics"]
    training = section(metrics, "training")
    dataset = section(metrics, "dataset")
    evaluation = section(metrics, "evaluation")
    benchmark = section(metrics, "benchmark")
    model = section(metrics, "model")
    resolution = records["resolution"]

    def raw(value):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value) else None

    scores = []
    for key, label, meaning in (
        ("precision", "Precisão", "Das caixas apontadas, quantas estavam certas"),
        ("recall", "Recall", "Dos resíduos anotados, quantos foram encontrados"),
        ("f1", "F1", "Equilíbrio entre precisão e recall"),
        ("map50", "AP50", "Qualidade da curva precisão × recall com IoU 0,50"),
        ("map50_95", "AP50-95", "Mesma medida, exigindo caixas mais justas"),
    ):
        scores.append({"key": key, "label": label, "meaning": meaning,
                       "value": raw(evaluation.get(key)),
                       "text": number(evaluation.get(key), 1, True)})
    device = str(benchmark.get("device", "Não registrado"))
    if device not in {"cpu", "Não registrado"}:
        device = str(training.get("gpu") or device)
    duration = raw(training.get("duration_seconds"))
    links = records["links"]
    return {
        "issues": records["issues"],
        "model": {"name": _text(model.get("name")),
                  "sha256": _text(model.get("sha256"))[:12]},
        "team": [{"name": _text(member.get("name"), "Não informado"),
                  "rm": _text(str(member.get("rm", "")), "Não informado")}
                 for member in records["team"] if isinstance(member, dict)],
        "links": {key: url for key, url in links.items()
                  if isinstance(url, str) and url.startswith("https://")},
        "dataset": {
            "source": _text(dataset.get("source")),
            "splits": [{"label": label, "images": raw(dataset.get(f"{key}_images")),
                        "images_text": number(dataset.get(f"{key}_images")),
                        "instances_text": number(dataset.get(f"{key}_instances"))}
                       for label, key in (("Treino", "train"), ("Validação", "val"),
                                          ("Teste", "test"))],
        },
        "training": [
            ["Arquitetura de origem", _text(training.get("base_model"))],
            ["Épocas concluídas", number(training.get("epochs_completed"))],
            ["Resolução de entrada", number(training.get("imgsz")) + " px"],
            ["Lote", number(training.get("batch"))],
            ["Seed", number(training.get("seed"))],
            ["GPU", _text(training.get("gpu"), "Não registrada")],
            ["Duração", number(duration / 60, 0) + " min" if duration else "Não registrada"],
        ],
        "experiments": [
            {"name": _text(item.get("name")), "epochs": number(item.get("epochs")),
             "batch": number(item.get("batch")),
             "map50": number(item.get("validation_map50"), 1, True)}
            for item in metrics.get("experiments", []) if isinstance(item, dict)
        ],
        "resolution": {
            "selected": number(resolution.get("selected_imgsz")),
            "candidates": [{"imgsz": number(item.get("imgsz")),
                            "map50": number(item.get("map50"), 1, True),
                            "map50_95": number(item.get("map50_95"), 1, True)}
                           for item in resolution.get("candidates", []) if isinstance(item, dict)],
        },
        "evaluation": {
            "images": number(dataset.get("test_images")),
            "instances": number(dataset.get("test_instances")),
            "confidence": number(evaluation.get("confidence"), 2),
            "iou": number(evaluation.get("iou"), 2),
            "target": TARGET,
            "target_text": number(TARGET, 2),
            "below_target": any(score["value"] is not None and score["value"] < TARGET
                                for score in scores if score["key"] in {"precision", "recall", "map50"}),
            "scores": scores,
            "counts": [
                {"label": "Acertos", "text": number(evaluation.get("true_positives"))},
                {"label": "Falsos positivos", "text": number(evaluation.get("false_positives"))},
                {"label": "Não encontrados", "text": number(evaluation.get("false_negatives"))},
            ],
        },
        "benchmark": [
            ["Dispositivo", device],
            ["Imagens medidas", number(benchmark.get("images"))],
            ["Taxa", number(benchmark.get("fps"), 1) + " imagens/s"],
            ["Latência mediana", number(benchmark.get("p50_ms"), 1) + " ms"],
            ["Latência p95", number(benchmark.get("p95_ms"), 1) + " ms"],
        ],
        "errors": [
            {"image": row["image"],
             "true_positives": number(row.get("true_positives")),
             "false_positives": number(row.get("false_positives")),
             "false_negatives": number(row.get("false_negatives")),
             "observation": _text(row.get("observation", row.get("description")), "Não registrada."),
             "hypothesis": _text(row.get("hypothesis"), "Não registrada."),
             "correction": _text(row.get("proposed_correction"), "Não registrada."),
             "figure": (f"/figures/{FIGURE_FOR_ERROR[row['image']]}"
                        if FIGURE_FOR_ERROR.get(row["image"]) in figures
                        and error_figure(row["image"], root) else None)}
            for row in metrics.get("errors", [])
            if isinstance(row, dict) and isinstance(row.get("image"), str)
        ],
        "limitations": [str(item) for item in metrics.get("limitations", []) if isinstance(item, str)],
        "figures": sorted(figures),
    }
