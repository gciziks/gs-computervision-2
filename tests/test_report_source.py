import json
from pathlib import Path


def test_report_word_limits():
    source = json.loads(Path("docs/RELATORIO.json").read_text(encoding="utf-8"))
    assert len(source["abstract"].split()) == 250
    assert len(source["objective"].split()) == 50
