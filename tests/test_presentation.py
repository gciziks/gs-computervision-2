import json

import pytest

from app.presentation import error_details, load_records, number


def test_missing_or_invalid_records_are_not_fabricated(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text("[]", encoding="utf-8")
    records = load_records(tmp_path)
    assert records["metrics"] == {}
    assert records["team"] == []
    assert "reports/metrics.json" in records["issues"]
    assert len(records["issues"]) == 4


def test_recorded_metrics_are_read_without_recalculation(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    metrics = {"evaluation": {"precision": 0.5423076923, "confidence": 0.35}}
    (reports / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert load_records(tmp_path)["metrics"] == metrics


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), "0.5"])
def test_unavailable_numbers_are_explicit(value):
    assert number(value) == "Não registrado"


def test_portuguese_number_format():
    assert number(0.5423076923, 1, True) == "54,2%"
    assert number(1854) == "1.854"


def test_error_figures_are_limited_to_reviewed_assets(tmp_path):
    assets = tmp_path / "assets/presentation"
    assets.mkdir(parents=True)
    figure = assets / "falha-000115.jpg"
    figure.touch()
    metrics = {"errors": [
        {"image": "000115.jpg", "true_positives": 0, "false_negatives": 3},
        {"image": "000113.jpg"},
        {"image": "../../private.jpg"},
    ]}
    text, path = error_details(metrics, "000115.jpg", tmp_path)
    assert "**Objetos não recuperados:** 3" in text
    assert path == str(figure)
    assert error_details(metrics, "000113.jpg", tmp_path)[1] is None
    assert error_details(metrics, "../../private.jpg", tmp_path)[1] is None
    assert error_details(metrics, "missing.jpg", tmp_path)[1] is None
