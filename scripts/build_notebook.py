"""Monta e, opcionalmente, executa o caderno de análise dos artefatos reais."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build_notebook(output: Path, execute: bool = False) -> Path:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    notebook.cells = [
        markdown("# OlhoLimpo: dados, treinamento e avaliação\n\n"
                 "Este caderno consulta os artefatos preservados da execução. Não repete o treinamento nem altera os conjuntos. "
                 "As saídas só são preenchidas quando as células são executadas. O README contém os comandos para refazer o pipeline completo."),
        code("from pathlib import Path\nimport csv\nimport json\nimport hashlib\n\n"
             "root = Path.cwd()\n"
             "while not (root / 'pyproject.toml').exists() and root != root.parent:\n    root = root.parent\n"
             "metrics_path = root / 'reports/metrics.json'\n"
             "if not metrics_path.is_file():\n    raise FileNotFoundError('Execute a avaliação antes deste caderno.')\n"
             "metrics = json.loads(metrics_path.read_text(encoding='utf-8'))\n"
             "print('Artefato:', metrics_path.relative_to(root))\n"
             "print('SHA-256:', hashlib.sha256(metrics_path.read_bytes()).hexdigest())"),
        markdown("## Dados e divisão\n\nA classe agregada é `residuo`. As contagens devem corresponder aos manifestos efetivamente usados, "
                 "sem contar aumentos como novas imagens independentes. A separação garante arquivos distintos e grupos encontrados por dHash; "
                 "não comprova independência de cenas ou objetos físicos. Os 778 grupos possuem uma imagem cada e os metadados são incompletos."),
        code("print(json.dumps(metrics.get('dataset', {}), indent=2, ensure_ascii=False))\n"
             "manifest_dir = root / 'data/manifests'\n"
             "print('Manifestos:', [p.name for p in sorted(manifest_dir.glob('*')) if p.is_file()])"),
        markdown("## Treinamento customizado\n\nConfiguração e identificação do checkpoint. A leitura dos registros não significa que o treino foi executado por este notebook."),
        code("print(json.dumps(metrics.get('training', {}), indent=2, ensure_ascii=False))\n"
             "print(json.dumps(metrics.get('model', {}), indent=2, ensure_ascii=False))"),
        code("declared_log = metrics.get('training', {}).get('results_csv')\n"
             "candidates = [root / declared_log] if declared_log else (sorted((root / 'runs').rglob('results.csv')) if (root / 'runs').exists() else [])\n"
             "if len(candidates) == 1 and candidates[0].is_file():\n"
             "    log_path = candidates[0]\n"
             "    with log_path.open(encoding='utf-8-sig', newline='') as stream:\n"
             "        history = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]\n"
             "    print('Log consultado:', log_path.relative_to(root))\n"
             "    print('Épocas registradas:', len(history))\n"
             "    print('Última linha:', history[-1] if history else {})\n"
             "else:\n"
             "    print('Defina training.results_csv nas métricas para identificar o log desta execução. Não foi escolhido um treino arbitrariamente.')\n"
             "    print('Arquivos encontrados:', [str(p.relative_to(root)) for p in candidates])"),
        markdown("## Avaliação reservada\n\nPrecisão, recall e AP pertencem ao protocolo indicado nas métricas. "
                 "O teste não deve orientar escolha de checkpoint ou limiar. Confiança não é probabilidade calibrada."),
        code("print(json.dumps(metrics['evaluation'], indent=2, ensure_ascii=False))\n"
             "print(json.dumps(metrics.get('benchmark', {}), indent=2, ensure_ascii=False))"),
        markdown("## Evidências visuais e erros\n\nExemplos de acertos e falhas devem manter sua identificação de origem. "
                 "Cenas controladas não substituem teste em campo."),
        code("from io import BytesIO\nfrom PIL import Image as PILImage\nfrom IPython.display import Image, display\n\n"
             "figures = [{'path': f} if isinstance(f, str) else f for f in metrics.get('figures', [])]\n"
             "examples = [f for f in figures if 'exemplo-' in f['path']][:2]\n"
             "curves = [f for f in figures if 'BoxPR_curve' in f['path']][:1]\n"
             "for item in examples + curves:\n"
             "    path = root / item['path']\n"
             "    with PILImage.open(path) as original:\n"
             "        preview = original.convert('RGB')\n"
             "        preview.thumbnail((960, 960))\n"
             "        buffer = BytesIO()\n"
             "        preview.save(buffer, format='JPEG', quality=85, optimize=True)\n"
             "    print(item.get('caption', path.name))\n"
             "    display(Image(data=buffer.getvalue(), format='jpeg', width=760))\n"
             "print('Pré-visualizações reduzidas; evidências completas preservadas nos arquivos de origem e no PDF.')\n"
             "print(json.dumps(metrics.get('errors', []), indent=2, ensure_ascii=False))"),
        markdown("## Interpretação\n\n"
                 "Conferir objetos pequenos, oclusão e fundos carregados. Uma embalagem em uso pode ter aparência igual à de uma descartada. "
                 "Ausência de caixa não comprova que o ambiente esteja limpo. Contagens por quadro não equivalem a objetos físicos únicos. "
                 "Novas melhorias exigem uma validação independente quando o teste anterior já tiver orientado decisões."),
    ]
    if execute:
        from nbclient import NotebookClient

        notebook = NotebookClient(notebook, timeout=180, kernel_name="python3",
                                  resources={"metadata": {"path": str(ROOT)}}).execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/avaliacao.ipynb")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(build_notebook(args.output, args.execute))


if __name__ == "__main__":
    main()
