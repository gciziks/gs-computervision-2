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
    # As figuras revisadas pertencem aos artefatos da apresentação, não a um caminho recebido da interface.
    approved = {"000115.jpg": root / "assets/presentation/falha-000115.jpg"}
    figure = approved.get(image)
    if figure is not None and figure.is_file():
        return text, str(figure)
    return text + "\n\nA fotografia deste caso não foi incluída na apresentação. Os resultados quantitativos permanecem registrados.", None


def build_presentation(root: Path):
    import gradio as gr

    records = load_records(root)
    metrics = records["metrics"]
    training = section(metrics, "training")
    dataset = section(metrics, "dataset")
    evaluation = section(metrics, "evaluation")
    benchmark = section(metrics, "benchmark")

    def table(rows, headers=("Item", "Registro")):
        def line(values):
            return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ")
                                      for value in values) + " |"

        return gr.Markdown("\n".join([line(headers), line(["---"] * len(headers)),
                                      *(line(row) for row in rows)]))

    def plot(relative, label):
        path = root / relative
        if path.is_file():
            gr.Image(value=str(path), label=label, interactive=False, height=420)
        else:
            gr.Markdown(f"**{label}:** arquivo não disponível. Gere os artefatos do treinamento para exibir esta figura.")

    with gr.Column(elem_id="presentation"):
        gr.Markdown("## Apresentação do OlhoLimpo\n"
                    "Equipe, metodologia e resultados da execução registrada. Use as seções abaixo durante a gravação.")
        model = section(metrics, "model")
        gr.Markdown(f"**Modelo avaliado:** {model.get('name', 'Não registrado')} · "
                    f"**SHA-256:** `{str(model.get('sha256', 'Não registrado'))[:12]}`")
        if records["issues"]:
            gr.Markdown("**Alguns registros estão indisponíveis ou inválidos:** "
                        + ", ".join(f"`{path}`" for path in records["issues"])
                        + ". Os campos correspondentes aparecem como não registrados.")
        with gr.Tabs():
            with gr.Tab("Projeto"):
                gr.Markdown("### O problema e o objetivo\n"
                            "Localizar resíduos visíveis em imagens e vídeos de espaços públicos. "
                            "O OlhoLimpo destaca regiões candidatas e permite conferir e exportar as detecções.\n\n"
                            "O detector identifica aparência visual; não determina intenção de descarte, "
                            "reciclabilidade ou risco sanitário.")
                gr.Markdown("### Equipe")
                members = [[str(member.get("name", "Não informado")), "RM " + str(member.get("rm", "Não informado"))]
                           for member in records["team"] if isinstance(member, dict)]
                if members:
                    table(members, ("Integrante", "RM"))
                else:
                    gr.Markdown("Integrantes não registrados. Preencha `deliverables/team.json`.")
                gr.Markdown("### Como funciona\n"
                            "Imagem, vídeo ou câmera → preparação do quadro → detector customizado → "
                            "caixas e confiança → visualização e exportação.\n\n"
                            "As abas **Imagem** e **Vídeo** mostram a aplicação em uso. "
                            "A câmera é executada pelo comando de webcam descrito em **Reprodução**.")
                gr.Markdown("### Ferramentas\n"
                            "Python e PyTorch executam o modelo; Ultralytics fornece o treinamento YOLO. "
                            "OpenCV processa imagens, vídeos e webcam. Gradio fornece esta interface local. "
                            "O TACO fornece as imagens e caixas usadas no ajuste supervisionado.")
            with gr.Tab("Treinamento"):
                gr.Markdown("### Treinamento customizado\n"
                            "O modelo parte de pesos pré-treinados e recebe ajuste supervisionado para a classe `residuo`. "
                            "As caixas do TACO são convertidas de COCO para YOLO. "
                            "O checkpoint e a confiança mínima são selecionados pela validação.")
                table([
                    ["Arquitetura de origem", str(training.get("base_model", "Não registrado"))],
                    ["Épocas concluídas", number(training.get("epochs_completed"))],
                    ["Resolução de treinamento", number(training.get("imgsz")) + " px"],
                    ["Lote", number(training.get("batch"))],
                    ["Seed", number(training.get("seed"))],
                    ["GPU do treinamento", str(training.get("gpu") or "Não registrada")],
                ])
                gr.Markdown("### Separação dos dados")
                table([[label, number(dataset.get(f"{split}_images")), number(dataset.get(f"{split}_instances"))]
                       for label, split in (("Treino", "train"), ("Validação", "val"), ("Teste", "test"))],
                      ("Conjunto", "Imagens", "Objetos anotados"))
                gr.Markdown("Aumentos são aplicados somente ao treino. Os conjuntos não compartilham arquivos ou "
                            "grupos encontrados pelo hash perceptual. Os metadados não permitem comprovar "
                            "independência entre todas as cenas ou objetos físicos.")
                gr.Markdown("### Curvas da execução real\n"
                            "Estas curvas pertencem ao treino e à validação. Os resultados do teste reservado estão na seção **Resultados**.")
                experiments = metrics.get("experiments", [])
                if experiments:
                    table([[item.get("name", "Não registrado"), number(item.get("epochs")),
                            number(item.get("batch")), number(item.get("validation_map50"), 1, True)]
                           for item in experiments if isinstance(item, dict)],
                          ("Execução", "Épocas", "Lote", "AP50 na validação"))
                    gr.Markdown("A referência inicial e o ajuste completo usam a mesma validação. "
                                "Épocas e lote mudaram; a comparação não isola o efeito de uma única variável.")
                plot("reports/training/results.png", "Perdas e métricas ao longo das épocas")
                with gr.Accordion("Precisão, recall e confiança na validação", open=False):
                    with gr.Row():
                        plot("reports/training/BoxPR_curve.png", "Precisão × recall — validação")
                        plot("reports/training/BoxF1_curve.png", "F1 × confiança — validação")
                resolution = records["resolution"]
                if resolution.get("candidates"):
                    gr.Markdown("### Escolha da resolução pela validação")
                    table([[number(item.get("imgsz")), number(item.get("map50"), 1, True),
                            number(item.get("map50_95"), 1, True)] for item in resolution["candidates"]
                           if isinstance(item, dict)], ("Entrada em pixels", "AP50", "AP50-95"))
                    gr.Markdown(f"**Entrada selecionada:** {number(resolution.get('selected_imgsz'))} px. "
                                "A comparação foi feita na validação, sem usar o teste para decidir.")
            with gr.Tab("Resultados"):
                gr.Markdown("### Avaliação no teste reservado\n"
                            f"**{number(dataset.get('test_images'))} imagens** e "
                            f"**{number(dataset.get('test_instances'))} objetos anotados**. "
                            f"Confiança mínima **{number(evaluation.get('confidence'), 2)}** e "
                            f"IoU de pareamento **{number(evaluation.get('iou'), 2)}**.\n\n"
                            "Estes são os resultados registrados da avaliação. Alterar o controle de confiança "
                            "nas abas Imagem e Vídeo não recalcula esta tabela.")
                table([[label, number(evaluation.get(key), 1, True)] for label, key in (
                    ("Precisão", "precision"), ("Recall", "recall"), ("F1", "f1"),
                    ("AP50", "map50"), ("AP50-95", "map50_95"))], ("Métrica", "Valor medido"))
                table([[label, number(evaluation.get(key))] for label, key in (
                    ("Acertos", "true_positives"), ("Falsos positivos", "false_positives"),
                    ("Objetos não recuperados", "false_negatives"))])
                gr.Markdown("Precisão indica a proporção de detecções corretas; recall indica a fração de objetos "
                            "anotados recuperados. AP resume a curva de precisão e recall. A confiança não é "
                            "uma probabilidade calibrada de acerto.")
                if evaluation and any(isinstance(evaluation.get(key), (int, float)) and evaluation[key] < .75
                                      for key in ("precision", "recall", "map50")):
                    gr.Markdown("**As metas internas de acurácia de 0,75 não foram atingidas.** "
                                "Os resultados não sustentam uma alegação de validação operacional.")
                gr.Markdown("### Desempenho medido")
                device = str(benchmark.get("device", "Não registrado"))
                if device not in {"cpu", "Não registrado"}:
                    device = str(training.get("gpu") or device)
                table([
                    ["Dispositivo", device],
                    ["Imagens medidas após aquecimento", number(benchmark.get("images"))],
                    ["Taxa de processamento", number(benchmark.get("fps"), 2) + " imagens/s"],
                    ["Latência mediana", number(benchmark.get("p50_ms"), 2) + " ms"],
                    ["Latência p95", number(benchmark.get("p95_ms"), 2) + " ms"],
                ])
                gr.Markdown("O benchmark inclui inferência e anotação sobre imagens já decodificadas. "
                            "Não inclui captura de câmera, leitura de arquivo ou exibição. "
                            "Esse valor não equivale à velocidade completa da webcam ou da interface.")
            with gr.Tab("Análise de erros"):
                gr.Markdown("### Casos do teste\n"
                            "As observações abaixo se referem às imagens avaliadas. As causas sugeridas são "
                            "hipóteses visuais; as correções propostas ainda precisam de novos experimentos.")
                choices = [row["image"] for row in metrics.get("errors", [])
                           if isinstance(row, dict) and isinstance(row.get("image"), str)]
                initial = "000115.jpg" if "000115.jpg" in choices else (choices[0] if choices else None)
                selector = gr.Dropdown(choices=choices, value=initial, label="Imagem do caso", interactive=bool(choices))
                text, figure = error_details(metrics, initial, root)
                details = gr.Markdown(text)
                example = gr.Image(value=figure, label="Caixas de referência em azul; predições em verde", interactive=False, height=400)
                selector.change(lambda image: error_details(metrics, image, root), inputs=selector,
                                outputs=[details, example], api_visibility="private", queue=False)
            with gr.Tab("Reprodução"):
                gr.Markdown("### Código, pesos e instruções")
                links = records["links"]
                available = [(label, links.get(key)) for label, key in (
                    ("Código no GitHub", "repository"), ("Relatório técnico", "report"), ("Checkpoint", "weights"))]
                gr.Markdown(" · ".join(f"[{label}]({url})" for label, url in available
                                       if isinstance(url, str) and url.startswith("https://"))
                            or "Links não registrados em `deliverables/links.json`.")
                gr.Markdown("### Abrir a aplicação")
                gr.Code("python -m app.main --device cpu", language="shell", interactive=False)
                gr.Markdown("### Usar a webcam local\n"
                            "A câmera exige dispositivo físico e permissão no sistema. "
                            "Na janela de captura, Q encerra e S salva um quadro.")
                gr.Code("olholimpo webcam --device 0", language="shell", interactive=False)
                with gr.Accordion("Refazer preparação, treino e avaliação", open=False):
                    gr.Code("python scripts/prepare_data.py\npython scripts/train.py --device 0\npython scripts/evaluate.py --device 0",
                            language="shell", interactive=False)
                gr.Markdown("### Limites e próximos passos\n"
                            "- Objetos pequenos, oclusão e fundos carregados podem causar erros.\n"
                            "- Não houve teste com negativos independentes nem ensaio físico da webcam nesta avaliação.\n"
                            "- Imagens do mesmo local em outros ângulos podem permanecer em conjuntos diferentes.\n"
                            "- Próximos experimentos: ampliar dados, revisar anotações e coletar novos testes independentes.\n\n"
                            "Os registros são lidos dos arquivos de equipe, métricas e treinamento do projeto ao iniciar a aplicação.")
