"""Gera o relatório técnico a partir das métricas e evidências da execução."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#142D32")
GREEN = colors.HexColor("#21604B")
MUTED = colors.HexColor("#52646A")
PALE = colors.HexColor("#EEF3EF")
LABELS = {
    "base_model": "Pesos de origem", "epochs_requested": "Épocas solicitadas",
    "epochs_completed": "Épocas concluídas", "batch": "Lote", "imgsz": "Resolução de entrada",
    "seed": "Seed", "duration_seconds": "Duração (s)", "gpu": "GPU", "torch": "PyTorch",
    "ultralytics": "Ultralytics", "results_csv": "Log do treinamento", "name": "Modelo",
    "sha256": "SHA-256 dos pesos", "weights": "Checkpoint", "source": "Fonte",
    "train_images": "Imagens de treino", "val_images": "Imagens de validação",
    "test_images": "Imagens de teste", "train_instances": "Instâncias de treino",
    "val_instances": "Instâncias de validação", "test_instances": "Instâncias de teste",
    "annotation_sha256": "SHA-256 das anotações", "independent_negatives": "Negativos independentes",
    "note": "Observação", "precision": "Precisão", "recall": "Recall", "f1": "F1",
    "true_positives": "Verdadeiros positivos", "false_positives": "Falsos positivos",
    "false_negatives": "Falsos negativos", "map50": "AP50", "map50_95": "AP50-95",
    "confidence": "Limiar de confiança", "iou": "IoU do pareamento",
    "selection_split": "Conjunto de seleção", "evaluation_split": "Conjunto avaliado",
    "ap_note": "Protocolo", "device": "Dispositivo", "images": "Imagens medidas",
    "warmup": "Aquecimento", "p50_ms": "Latência p50 (ms)", "p95_ms": "Latência p95 (ms)",
    "fps": "Imagens processadas/s", "scope": "Escopo da medição",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def word_counts(source: dict) -> dict[str, int]:
    counts = {key: len(source[key].split()) for key in ("abstract", "objective")}
    if counts != {"abstract": 250, "objective": 50}:
        raise ValueError(f"Contagem incorreta de palavras: {counts}")
    return counts


def display(value) -> str:
    if value is None or value == "":
        return "Não registrado"
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "Não registrado"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def section_data(metrics: dict, name: str) -> dict:
    value = metrics.get(name, {})
    return value if isinstance(value, dict) else {}


def make_styles():
    font_regular = Path("C:/Windows/Fonts/arial.ttf")
    font_bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if font_regular.exists() and font_bold.exists():
        pdfmetrics.registerFont(TTFont("ReportFont", str(font_regular)))
        pdfmetrics.registerFont(TTFont("ReportFont-Bold", str(font_bold)))
        pdfmetrics.registerFontFamily("ReportFont", normal="ReportFont", bold="ReportFont-Bold")
        font = "ReportFont"
        bold = "ReportFont-Bold"
    else:
        font, bold = "Helvetica", "Helvetica-Bold"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyTextPT", fontName=font, fontSize=10, leading=15,
                              textColor=INK, spaceAfter=9, alignment=TA_JUSTIFY,
                              allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle("TitlePT", fontName=bold, fontSize=27, leading=32,
                              textColor=INK, spaceAfter=16))
    styles.add(ParagraphStyle("HeadingPT", fontName=bold, fontSize=16, leading=21,
                              textColor=GREEN, spaceBefore=14, spaceAfter=10,
                              keepWithNext=True))
    styles.add(ParagraphStyle("SubheadingPT", fontName=bold, fontSize=11, leading=15,
                              textColor=GREEN, spaceBefore=8, spaceAfter=5,
                              keepWithNext=True))
    styles.add(ParagraphStyle("ErrorPT", fontName=font, fontSize=9, leading=13,
                              textColor=INK, spaceAfter=6))
    styles.add(ParagraphStyle("SmallPT", fontName=font, fontSize=8, leading=11,
                              textColor=MUTED, spaceAfter=7, wordWrap="CJK"))
    styles.add(ParagraphStyle("TablePT", fontName=font, fontSize=8.5, leading=12,
                              textColor=INK, wordWrap="CJK"))
    return styles


def build_report(metrics_path: Path, output: Path, render: bool = False) -> Path:
    if not metrics_path.is_file():
        raise FileNotFoundError("Execute a avaliação e gere reports/metrics.json antes do relatório.")
    metrics = load_json(metrics_path)
    if not isinstance(metrics, dict) or not section_data(metrics, "evaluation"):
        raise ValueError("As métricas precisam conter a seção evaluation com resultados reais.")
    source = load_json(ROOT / "docs/RELATORIO.json")
    counts = word_counts(source)
    team = load_json(ROOT / "deliverables/team.json")
    links = load_json(ROOT / "deliverables/links.json")
    styles = make_styles()
    story = []

    def paragraph(text, style="BodyTextPT"):
        text = str(text).replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])

    def heading(title):
        story.append(paragraph(title, "HeadingPT"))

    def prose(lines):
        story.extend(paragraph(line) for line in lines)

    def table(rows, header=("Item", "Registro")):
        cells = [[paragraph(value, "TablePT") for value in header]]
        cells.extend([[paragraph(display(value), "TablePT") for value in row] for row in rows])
        result = Table(cells, colWidths=[5.0 * cm, 11.2 * cm], repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PALE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, GREEN),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#D9E2DD")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend([result, Spacer(1, 10)])

    def metric_rows(values):
        return [(LABELS.get(key, key), value) for key, value in values.items()]

    story.append(paragraph(source["subtitle"].upper(), "SmallPT"))
    story.append(paragraph(source["title"], "TitlePT"))
    if team:
        for member in team:
            if isinstance(member, dict):
                name = member.get("name") or member.get("nome") or "Nome não informado"
                rm = member.get("rm")
                story.append(paragraph(f"{name}" + (f" | RM {rm}" if rm else "")))
            else:
                story.append(paragraph(member))
    else:
        story.append(paragraph("Identificação dos integrantes pendente."))
    link_parts = []
    for label, key in (("Código no GitHub", "repository"), ("Vídeo demonstrativo", "video"), ("Checkpoint do modelo", "weights")):
        if links.get(key):
            url = str(links[key])
            link_parts.append(f'<link href="{escape(url, {chr(34): "&quot;"})}" color="#21604B">{label}</link>')
    if link_parts:
        story.append(Paragraph(" | ".join(link_parts), styles["SmallPT"]))
    missing = [label for label, key in (("repositório", "repository"), ("vídeo", "video")) if not links.get(key)]
    if missing:
        story.append(paragraph("Links de entrega pendentes: " + ", ".join(missing) + ".", "SmallPT"))
    heading("Resumo")
    prose([source["abstract"]])
    story.append(paragraph(f"Contagem: {counts['abstract']} palavras.", "SmallPT"))
    heading("Objetivo")
    prose([source["objective"]])
    story.append(paragraph(f"Contagem: {counts['objective']} palavras.", "SmallPT"))

    story.append(PageBreak())
    heading("1. Introdução")
    prose(source["introduction"])
    heading("2. Ferramentas e fundamentos")
    for name, description in source["tools"]:
        story.append(KeepTogether([paragraph(name, "SubheadingPT"), paragraph(description)]))

    story.append(PageBreak())
    heading("3. Metodologia")
    for index, step in enumerate(source["methodology"], 1):
        prose([f"{index}. {step}"])
    heading("Configuração da execução")
    training = section_data(metrics, "training")
    model = section_data(metrics, "model")
    table(metric_rows(training) or [("Treinamento", "Consultar configuração e logs junto ao checkpoint.")])
    table(metric_rows(model) or [("Modelo", "Identificação não registrada nas métricas.")])

    heading("4. Dados e separação dos conjuntos")
    prose(["As contagens abaixo pertencem aos arquivos efetivamente preparados nesta execução. A classe agregada é residuo. Os manifestos preservam os identificadores e o mapeamento dos rótulos originais."])
    dataset = section_data(metrics, "dataset")
    table(metric_rows(dataset) or [("Dataset", "Consultar data/manifests e docs/DATA-CARD.md.")])
    prose([
        "Aumentos não são novas capturas independentes. A separação garante arquivos distintos e grupos "
        "identificados pelo hash perceptual dHash, não independência comprovada de cenas físicas. Os 778 grupos "
        "resultantes possuem uma imagem cada; os metadados não identificam completamente local, sessão e objeto. "
        "Fotografias próximas no TACO podem mostrar o mesmo lugar sob ângulos diferentes, mesmo quando o hash não as agrupa.",
        "Sem teste externo próprio, os resultados são uma avaliação interna da distribuição selecionada do TACO. "
        "Não foi presumida validação operacional em campo."
    ])
    heading("5. Resultados quantitativos")
    evaluation = section_data(metrics, "evaluation")
    table(metric_rows(evaluation))
    precision = float(evaluation.get("precision", 0))
    recall = float(evaluation.get("recall", 0))
    ap50 = float(evaluation.get("map50", 0))
    fn = int(evaluation.get("false_negatives", 0))
    tp = int(evaluation.get("true_positives", 0))
    interpretation = [
        f"No teste reservado, a precisão foi {precision:.1%} e o recall foi {recall:.1%}, "
        f"com AP50 de {ap50:.3f}. No ponto de operação de confiança {evaluation.get('confidence')} "
        f"e IoU {evaluation.get('iou')}, {fn} das {tp + fn} instâncias anotadas ficaram sem correspondência. "
        "Esse volume de falsos negativos limita o uso do detector para inspeção sem revisão dos registros.",
        "Precisão, recall e AP50 não atingiram as metas internas de 0,75 definidas no planejamento. "
        "Essas metas são objetivos de engenharia, não limiares fixados no enunciado. Os resultados "
        "demonstram o treinamento e permitem analisar falhas; não sustentam uma alegação de acurácia operacional."
    ]
    story.extend(KeepTogether([paragraph(text)]) for text in interpretation)
    prose(source["metric_notes"])

    resolution_path = ROOT / "reports/resolution-selection.json"
    if resolution_path.is_file():
        selection = load_json(resolution_path)
        heading("Escolha da resolução pela validação")
        rows = []
        for candidate in selection.get("candidates", []):
            rows.append((f"Entrada {candidate['imgsz']}",
                         f"AP50 {candidate['map50']:.4f}; AP50-95 {candidate['map50_95']:.4f}"))
        table(rows)
        prose([f"A regra registrada foi: {selection.get('rule', 'Consultar artefato de seleção.')} "
               f"A resolução selecionada foi {selection.get('selected_imgsz')}. "
               "A entrada de 960 aumentou levemente a AP50, mas não melhorou a AP50-95 na validação; "
               "por isso foi mantida a entrada de 640. A escolha não usou o teste reservado."])

    heading("6. Desempenho e exemplos")
    benchmark = section_data(metrics, "benchmark")
    table(metric_rows(benchmark) or [("Benchmark", "Medição não registrada; não há alegação de FPS ou latência.")])
    prose([
        f"O valor de {float(benchmark.get('fps', 0)):.2f} imagens/s foi medido na GPU com imagens "
        "pré-decodificadas, incluindo inferência e anotação. Não inclui leitura de arquivo, captura de câmera "
        "ou exibição. Portanto, não equivale a throughput de webcam nem comprova a meta de captura até exibição.",
        "A aplicação foi testada com os pesos reais em CPU e GPU. A câmera física não foi ativada nesta execução; "
        "o ensaio de webcam e a validação urbana permanecem pendentes. Uma sequência criada a partir de imagens "
        "estáticas é evidência de processamento contínuo, não uma captura real de vídeo."
    ])
    figures = metrics.get("figures", [])
    if not isinstance(figures, list):
        figures = []
    figure_count = 0
    for figure in figures:
        item = {"path": figure} if isinstance(figure, str) else figure
        image_path = ROOT / item.get("path", "")
        if not image_path.is_file():
            raise FileNotFoundError(f"Figura referenciada não encontrada: {image_path}")
        figure_count += 1
        from PIL import Image as PILImage

        with PILImage.open(image_path) as original:
            preview = original.convert("RGB")
            preview.thumbnail((1800, 1800))
            buffer = BytesIO()
            preview.save(buffer, format="JPEG", quality=90, optimize=True)
        buffer.seek(0)
        image = Image(buffer)
        ratio = min(16.2 * cm / image.imageWidth, 16.0 * cm / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * ratio, image.imageHeight * ratio
        story.append(KeepTogether([Spacer(1, 8), image, paragraph(f"Figura {figure_count}. {item.get('caption', image_path.stem)}", "SmallPT")]))
    if not figures:
        prose(["Exemplos visuais ainda não foram associados às métricas. Este documento precisa dessas evidências antes da entrega final."])

    error_heading = paragraph("7. Análise de erros", "HeadingPT")
    errors = metrics.get("errors", [])
    if isinstance(errors, list) and errors:
        for index, error in enumerate(errors, 1):
            if isinstance(error, dict):
                case = [paragraph(f"Caso {index}: {error.get('image', 'imagem não identificada')}", "SubheadingPT")]
                if index == 1:
                    case.insert(0, error_heading)
                case.extend(paragraph(text, "ErrorPT") for text in [
                    f"Pareamento: acertos: {error.get('true_positives', 0)}; "
                    f"falsos positivos: {error.get('false_positives', 0)}; "
                    f"objetos não recuperados: {error.get('false_negatives', 0)}.",
                    "Observação: " + error.get("observation", error.get("description", "Não registrada.")),
                    "Hipótese: " + error.get("hypothesis", "Não registrada."),
                    "Correção proposta: " + error.get("proposed_correction", "Não registrada."),
                ])
                story.append(KeepTogether(case))
            else:
                if index == 1:
                    story.append(error_heading)
                prose([f"{index}. {error}"])
    else:
        story.append(error_heading)
        prose(["A análise individual de falhas ainda não foi registrada. Não foram inventados exemplos ou causas; completar esta seção a partir das predições antes da entrega final."])

    heading("8. Limitações")
    prose([
        "A avaliação é interna: não houve coleta própria, negativos independentes ou validação urbana. "
        "A câmera física não foi ativada. Cenas controladas ou sequências de imagens não substituem esses ensaios.",
        "Os conjuntos são distintos por arquivo e pelo agrupamento dHash aplicado. Isso não comprova que "
        "cenas ou objetos físicos sejam independentes entre os conjuntos; os metadados incompletos limitam essa verificação.",
        "Objetos pequenos, oclusão, desfoque e fundos carregados podem alterar o resultado. Objetos em uso "
        "podem ser confundidos com resíduos. O detector não determina intenção de descarte, propriedade, "
        "composição ou risco sanitário; ausência de caixa não comprova um ambiente limpo.",
        "As licenças das imagens apresentam metadados incompletos; consultar docs/DATA-CARD.md antes de "
        "redistribuir conteúdo. A confiança não é uma probabilidade calibrada e os resultados abaixo das metas "
        "internas não sustentam uma alegação de acurácia operacional."
    ])
    heading("9. Conclusão")
    prose([
        f"O detector foi ajustado por {training.get('epochs_completed')} épocas e avaliado em "
        f"{dataset.get('test_images')} imagens reservadas, com {dataset.get('test_instances')} instâncias anotadas. "
        f"A precisão de {precision:.1%}, o recall de {recall:.1%} e a AP50 de {ap50:.3f} ficaram abaixo das metas "
        "internas. A alta quantidade de objetos não recuperados impede tratar o resultado como inspeção confiável "
        "sem revisão humana. O treinamento customizado, as métricas e a demonstração local são reproduzíveis, "
        "mas não representam validação operacional.",
        "Os próximos passos são ampliar exemplos de objetos pequenos e cenas densas, revisar instâncias difíceis "
        "e coletar negativos independentes. Essas propostas partem da inspeção dos erros; sua eficácia ainda não "
        "foi comprovada. Como o teste já foi inspecionado, novas decisões deverão usar validação própria e um novo "
        "teste independente para sustentar conclusões posteriores. A webcam física e cenas urbanas próprias "
        "também precisam de ensaio."
    ])
    heading("10. Reprodução e referências")
    prose(["O README contém os comandos da preparação, treinamento, avaliação e aplicação. Os manifestos e o hash do checkpoint permitem identificar os artefatos desta execução. A gravação deve mostrar o mesmo modelo descrito aqui."])
    for label, url in source["references"]:
        story.append(Paragraph(f'{escape(label)}<br/><link href="{url}" color="#21604B">{url}</link>', styles["SmallPT"]))

    def page(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9E2DD"))
        canvas.line(2.4 * cm, 1.55 * cm, A4[0] - 2.4 * cm, 1.55 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.4 * cm, 1.1 * cm, "OlhoLimpo | Applied Computer Vision")
        canvas.drawRightString(A4[0] - 2.4 * cm, 1.1 * cm, str(document.page))
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=2.4 * cm,
                                 leftMargin=2.4 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                                 title=source["title"], author="; ".join(
                                     str(member.get("name", member.get("nome", ""))) if isinstance(member, dict) else str(member)
                                     for member in team))
    document.build(story, onFirstPage=page, onLaterPages=page)
    if render:
        import pymupdf

        render_dir = ROOT / "tmp/pdfs/olholimpo"
        render_dir.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(output) as pdf:
            for number, pdf_page in enumerate(pdf, 1):
                pdf_page.get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(render_dir / f"page-{number:02d}.png")
        print(f"Páginas renderizadas em {render_dir}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports/metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/olholimpo-relatorio.pdf")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    print(build_report(args.metrics, args.output, args.render))


if __name__ == "__main__":
    main()
