"""Gera o relatório técnico a partir das métricas e evidências da execução."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#1B2A24")
ACCENT = colors.HexColor("#2C674C")
MUTED = colors.HexColor("#5B6B63")
RULE = colors.HexColor("#C9D2CB")
WIDTH = A4[0] - 2 * 2.5 * cm
FIGURE_CAPTIONS = {
    "results.png": "Perdas e métricas por época, no treino e na validação.",
    "BoxPR_curve.png": "Precisão e recall na validação.",
    "BoxF1_curve.png": "F1 por limiar de confiança, na validação.",
    "comparacao-validacao.png": "AP50 na validação: referência com 1 época e lote 8, ajuste completo com 60 épocas e lote 16.",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def word_counts(source: dict) -> dict[str, int]:
    counts = {key: len(source[key].split()) for key in ("abstract", "objective")}
    if counts != {"abstract": 250, "objective": 50}:
        raise ValueError(f"Contagem incorreta de palavras: {counts}")
    return counts


def section_data(metrics: dict, name: str) -> dict:
    value = metrics.get(name, {})
    return value if isinstance(value, dict) else {}


def br(value, decimals=0, percent=False, suffix="") -> str:
    """Número no formato brasileiro, ou 'não registrado' quando ausente."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "não registrado"
    text = f"{value * 100 if percent else value:,.{decimals}f}"
    text = text.translate(str.maketrans({",": ".", ".": ","}))
    return text + ("%" if percent else "") + suffix


def count(value, singular: str, plural: str) -> str:
    return f"{br(value)} {singular if value == 1 else plural}"


NUMBER_WORDS = {2: "Duas", 3: "Três", 4: "Quatro", 5: "Cinco", 6: "Seis", 7: "Sete", 8: "Oito"}


def register_fonts() -> tuple[str, str, str, str]:
    fonts = Path("C:/Windows/Fonts")
    serif = {"regular": "georgia.ttf", "bold": "georgiab.ttf", "italic": "georgiai.ttf"}
    sans = {"regular": "segoeui.ttf", "bold": "segoeuib.ttf"}
    if all((fonts / name).is_file() for name in [*serif.values(), *sans.values()]):
        pdfmetrics.registerFont(TTFont("Body", str(fonts / serif["regular"])))
        pdfmetrics.registerFont(TTFont("Body-Bold", str(fonts / serif["bold"])))
        pdfmetrics.registerFont(TTFont("Body-Italic", str(fonts / serif["italic"])))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")
        pdfmetrics.registerFont(TTFont("Head", str(fonts / sans["regular"])))
        pdfmetrics.registerFont(TTFont("Head-Bold", str(fonts / sans["bold"])))
        pdfmetrics.registerFontFamily("Head", normal="Head", bold="Head-Bold")
        return "Body", "Body-Bold", "Head", "Head-Bold"
    return "Times-Roman", "Times-Bold", "Helvetica", "Helvetica-Bold"


def make_styles() -> dict[str, ParagraphStyle]:
    body, body_bold, head, head_bold = register_fonts()
    base = dict(fontName=body, fontSize=10.5, leading=16, textColor=INK, alignment=TA_LEFT,
                allowWidows=0, allowOrphans=0)
    return {
        "body": ParagraphStyle("body", spaceAfter=8, **base),
        "lead": ParagraphStyle("lead", **{**base, "fontSize": 10.5, "spaceAfter": 0}),
        "title": ParagraphStyle("title", fontName=head_bold, fontSize=25, leading=30, textColor=INK,
                                spaceAfter=6),
        "kicker": ParagraphStyle("kicker", fontName=head, fontSize=11, leading=14, textColor=MUTED,
                                 spaceAfter=18),
        "h1": ParagraphStyle("h1", fontName=head_bold, fontSize=15, leading=20, textColor=INK,
                             spaceBefore=18, spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("h2", fontName=head_bold, fontSize=11, leading=15, textColor=ACCENT,
                             spaceBefore=10, spaceAfter=3, keepWithNext=True),
        "meta": ParagraphStyle("meta", fontName=head, fontSize=9.5, leading=14, textColor=MUTED),
        "cell": ParagraphStyle("cell", fontName=head, fontSize=9.5, leading=13, textColor=INK),
        "cell_head": ParagraphStyle("cell_head", fontName=head_bold, fontSize=9, leading=12,
                                    textColor=MUTED),
        "num": ParagraphStyle("num", fontName=head, fontSize=9.5, leading=13, textColor=INK,
                              alignment=2),
        "num_head": ParagraphStyle("num_head", fontName=head_bold, fontSize=9, leading=12,
                                   textColor=MUTED, alignment=2),
        "caption": ParagraphStyle("caption", fontName=head, fontSize=8.5, leading=12,
                                  textColor=MUTED, spaceBefore=4, spaceAfter=12),
        "ref": ParagraphStyle("ref", fontName=body, fontSize=9.5, leading=14, textColor=INK,
                              spaceAfter=6, leftIndent=12, firstLineIndent=-12),
        "page": ParagraphStyle("page", fontName=head, fontSize=8.5, textColor=MUTED,
                               alignment=TA_CENTER),
        "_body_bold": body_bold,
        "_head": head,
    }


def clean(text) -> str:
    text = str(text).replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
    # A ressalva sobre hipóteses aparece uma vez no texto da seção, não em cada caso.
    return re.sub(r"; é uma hipótese visual, não uma explicação causal comprovada\.?$", ".", text)


def build_report(metrics_path: Path, output: Path, render: bool = False) -> Path:
    if not metrics_path.is_file():
        raise FileNotFoundError("Execute a avaliação e gere reports/metrics.json antes do relatório.")
    metrics = load_json(metrics_path)
    if not isinstance(metrics, dict) or not section_data(metrics, "evaluation"):
        raise ValueError("As métricas precisam conter a seção evaluation com resultados reais.")
    source = load_json(ROOT / "docs/RELATORIO.json")
    word_counts(source)
    team = load_json(ROOT / "deliverables/team.json")
    links = load_json(ROOT / "deliverables/links.json")
    summary_path = ROOT / "data/manifests/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    resolution_path = ROOT / "reports/resolution-selection.json"
    resolution = load_json(resolution_path) if resolution_path.is_file() else {}
    training = section_data(metrics, "training")
    model = section_data(metrics, "model")
    dataset = section_data(metrics, "dataset")
    evaluation = section_data(metrics, "evaluation")
    benchmark = section_data(metrics, "benchmark")
    styles = make_styles()
    story = []

    def para(text, style="body", markup=False):
        text = clean(text) if not markup else str(text)
        return Paragraph(text if markup else escape(text), styles[style])

    def heading(text, level="h1"):
        story.append(para(text, level))

    def table(header, rows, widths, numeric=()):
        cells = [[para(value, "num_head" if index in numeric else "cell_head")
                  for index, value in enumerate(header)]]
        cells += [[para(value, "num" if index in numeric else "cell")
                   for index, value in enumerate(row)] for row in rows]
        result = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
        result.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (-1, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return result

    figure_number = [0]

    def figure(path: Path, caption: str, width: float):
        if not path.is_file():
            raise FileNotFoundError(f"Figura referenciada não encontrada: {path}")
        from PIL import Image as PILImage

        with PILImage.open(path) as original:
            preview = original.convert("RGB")
            preview.thumbnail((1800, 1800))
            buffer = BytesIO()
            preview.save(buffer, format="JPEG", quality=88, optimize=True)
        buffer.seek(0)
        image = Image(buffer)
        ratio = min(width / image.imageWidth, 11.5 * cm / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * ratio, image.imageHeight * ratio
        image.hAlign = "LEFT"
        figure_number[0] += 1
        return [image, para(f"Figura {figure_number[0]}. {caption}", "caption")]

    # Capa, resumo e objetivo
    story.append(Spacer(1, 1.2 * cm))
    story.append(para(source["title"], "title"))
    story.append(para(source["subtitle"], "kicker"))
    members = [m for m in team if isinstance(m, dict)]
    if members:
        rows = [[para(str(m.get("name", "Nome não informado")), "cell"),
                 para(f"RM {m['rm']}" if m.get("rm") else "", "cell")] for m in members]
        names = Table(rows, colWidths=[7.5 * cm, 3 * cm], hAlign="LEFT")
        names.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("TOPPADDING", (0, 0), (-1, -1), 1),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
        story.append(names)
    available = [(label, links.get(key)) for label, key in (
        ("código", "repository"), ("vídeo da demonstração", "video"), ("pesos do modelo", "weights"))
        if isinstance(links.get(key), str) and links[key].startswith("https://")]
    if available:
        parts = [f'<link href="{escape(url, {chr(34): "&quot;"})}" color="#2C674C">{label}</link>'
                 for label, url in available]
        joined = ", ".join(parts[:-1]) + " e " + parts[-1] if len(parts) > 1 else parts[0]
        story.append(Spacer(1, 10))
        story.append(para("Disponíveis no repositório: " + joined + ".", "meta", markup=True))
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceAfter=4))
    heading("Resumo")
    story.append(para(source["abstract"]))
    heading("Objetivo")
    story.append(para(source["objective"]))
    story.append(PageBreak())

    # 1. Introdução
    heading("1. Introdução")
    story.extend(para(text) for text in source["introduction"])

    # 2. Ferramentas
    heading("2. Ferramentas")
    story.append(table(("Ferramenta", "Uso no projeto"), source["tools"], [4.2 * cm, WIDTH - 4.2 * cm]))

    # 3. Dados
    heading("3. Dados")
    splits = [("Treino", "train"), ("Validação", "val"), ("Teste", "test")]
    total_images = sum(dataset.get(f"{key}_images", 0) for _, key in splits)
    story.append(para(
        f"Das {br(summary.get('source_images'))} imagens do TACO, {br(total_images)} entraram no projeto, "
        f"com {br(summary.get('prepared_annotations'))} objetos anotados, cerca de três por imagem. "
        f"A divisão usou a seed {br(summary.get('seed', training.get('seed')))} e nenhum grupo de quase "
        "duplicatas aparece em mais de um conjunto."))
    story.append(table(
        ("Conjunto", "Imagens", "Objetos anotados", "Uso"),
        [[label, br(dataset.get(f"{key}_images")), br(dataset.get(f"{key}_instances")), use]
         for (label, key), use in zip(splits, (
             "ajuste dos pesos", "escolha de checkpoint, resolução e limiar", "avaliação final"))],
        [3 * cm, 2.2 * cm, 3.2 * cm, WIDTH - 8.4 * cm], numeric=(1, 2)))

    # 4. Metodologia
    heading("4. Metodologia")
    for title, text in source["methodology"]:
        story.append(para(title, "h2"))
        story.append(para(text))
    duration = training.get("duration_seconds")
    story.append(Spacer(1, 4))
    story.append(KeepTogether([table(("Parâmetro", "Valor"), [
        ["Pesos de origem", training.get("base_model", "não registrado")],
        ["Épocas", br(training.get("epochs_completed"))],
        ["Lote", br(training.get("batch"))],
        ["Resolução de entrada", br(training.get("imgsz"), suffix=" px")],
        ["Seed", br(training.get("seed"))],
        ["GPU", training.get("gpu", "não registrada")],
        ["Duração do treino", br(duration / 60 if isinstance(duration, (int, float)) else None, suffix=" min")],
        ["Bibliotecas", f"PyTorch {training.get('torch', '?')}, Ultralytics {training.get('ultralytics', '?')}"],
        ["Checkpoint (SHA-256)", str(model.get("sha256", "não registrado"))[:16]],
    ], [5 * cm, WIDTH - 5 * cm])]))

    # 5. Resultados
    heading("5. Resultados no teste reservado")
    tp, fp, fn = (int(evaluation.get(key, 0)) for key in
                  ("true_positives", "false_positives", "false_negatives"))
    intro = (para(
        f"O teste tem {br(dataset.get('test_images'))} imagens e {br(dataset.get('test_instances'))} objetos "
        f"anotados. A avaliação usou o limiar de confiança {br(evaluation.get('confidence'), 2)}, escolhido na "
        f"validação, e IoU mínima de {br(evaluation.get('iou'), 2)} para contar um acerto."))
    scores = table(("Métrica", "Valor"), [
        [label, br(evaluation.get(key), 1, True)] for label, key in (
            ("Precisão", "precision"), ("Recall", "recall"), ("F1", "f1"),
            ("AP50", "map50"), ("AP50-95", "map50_95"))], [3.6 * cm, 2.4 * cm], numeric=(1,))
    counts = table(("Pareamento", "Caixas"), [
        ["Acertos", br(tp)], ["Falsos positivos", br(fp)], ["Objetos não encontrados", br(fn)]],
        [4.6 * cm, 2 * cm], numeric=(1,))
    pair = Table([[scores, counts]], colWidths=[7.2 * cm, WIDTH - 7.2 * cm], hAlign="LEFT")
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(KeepTogether([intro, Spacer(1, 2), pair]))
    story.append(Spacer(1, 10))
    story.append(para(
        f"Das {br(tp + fn)} instâncias anotadas, o modelo encontrou {br(tp)} e deixou {br(fn)} sem "
        f"correspondência. Entre as {br(tp + fp)} caixas que propôs, {br(fp)} não correspondiam a nenhum "
        "objeto. Ou seja, quando o detector aponta algo ele acerta pouco mais da metade das vezes, mas deixa "
        "de ver a maioria dos objetos, como a análise de erros da seção 7 ilustra. Precisão, recall e AP50 "
        "ficaram abaixo da meta interna de 0,75 definida no planejamento."))
    story.append(para(source["metric_note"]))
    candidates = [item for item in resolution.get("candidates", []) if isinstance(item, dict)]
    if candidates:
        story.append(para("Escolha da resolução", "h2"))
        story.append(para(
            f"Antes de abrir o teste, a entrada de 960 px foi comparada à de 640 px na validação. A regra "
            f"registrada era adotar a maior apenas com ganho acima de 0,005 em AP50-95, o que não aconteceu, "
            f"e a resolução de {br(resolution.get('selected_imgsz'))} px foi mantida."))
        story.append(table(("Entrada", "AP50 (validação)", "AP50-95 (validação)"), [
            [br(item.get("imgsz"), suffix=" px"), br(item.get("map50"), 1, True),
             br(item.get("map50_95"), 1, True)] for item in candidates],
            [3 * cm, 3.6 * cm, 3.8 * cm], numeric=(1, 2)))
        story.append(Spacer(1, 10))

    figures = [item if isinstance(item, dict) else {"path": item} for item in metrics.get("figures", [])]
    by_name = {Path(item.get("path", "")).name: ROOT / item["path"] for item in figures if item.get("path")}
    if "results.png" in by_name:
        story.append(KeepTogether(figure(by_name["results.png"], FIGURE_CAPTIONS["results.png"], WIDTH)))
    curves = [name for name in ("BoxPR_curve.png", "BoxF1_curve.png") if name in by_name]
    if curves:
        half = (WIDTH - 0.6 * cm) / 2
        cells = [figure(by_name[name], FIGURE_CAPTIONS[name], half) for name in curves]
        grid = Table([cells], colWidths=[half + 0.6 * cm] * (len(cells) - 1) + [half], hAlign="LEFT")
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(grid)
    if "comparacao-validacao.png" in by_name:
        story.append(KeepTogether(figure(by_name["comparacao-validacao.png"],
                                         FIGURE_CAPTIONS["comparacao-validacao.png"], WIDTH * 0.62)))

    # 6. Velocidade
    device = str(benchmark.get("device", ""))
    device_name = training.get("gpu") if device not in {"", "cpu"} else "CPU"
    speed = (para(
        f"A velocidade foi medida em {br(benchmark.get('images'))} imagens já decodificadas, depois do "
        "aquecimento do modelo, contando inferência e desenho das caixas. Leitura do arquivo, captura da "
        "câmera e exibição ficam de fora, por isso o tempo percebido na aplicação é maior."))
    story.append(KeepTogether([para("6. Velocidade", "h1"), speed, table(("Dispositivo", "Latência mediana", "Latência p95", "Taxa"), [[
        device_name or "não registrado", br(benchmark.get("p50_ms"), 1, suffix=" ms"),
        br(benchmark.get("p95_ms"), 1, suffix=" ms"), br(benchmark.get("fps"), 1, suffix=" imagens/s")]],
        [5.4 * cm, 3.4 * cm, 3 * cm, WIDTH - 11.8 * cm], numeric=(1, 2, 3))]))
    story.append(Spacer(1, 8))
    story.append(para(
        "A aplicação também roda em CPU. Na demonstração, imagens e vídeos são enviados pelo navegador, "
        "as caixas aparecem sobre a imagem e os resultados podem ser exportados em PNG, JSON e CSV. A webcam "
        "usa um comando de terminal com janela do OpenCV."))

    # 7. Análise de erros
    errors = [item for item in metrics.get("errors", []) if isinstance(item, dict)]
    heading("7. Análise de erros")
    story.append(para(
        f"{NUMBER_WORDS.get(len(errors), br(len(errors)))} imagens do teste foram revisadas uma a uma. Nas figuras, as caixas azuis são as "
        "anotações e as verdes, as predições. As causas indicadas são leituras visuais e ainda não foram "
        "testadas em novos experimentos."))
    examples = {name.removeprefix("exemplo-").removesuffix(".png"): path
                for name, path in by_name.items() if name.startswith("exemplo-")}
    for error in errors:
        image = str(error.get("image", "imagem não identificada"))
        stem = Path(image).stem
        block = [para(f"Imagem {stem}", "h2"), para(
            f"{count(error.get('true_positives'), 'acerto', 'acertos')}, "
            f"{count(error.get('false_positives'), 'falso positivo', 'falsos positivos')} e "
            f"{count(error.get('false_negatives'), 'objeto não encontrado', 'objetos não encontrados')}.",
            "meta"), Spacer(1, 4)]
        if stem in examples:
            block += figure(examples[stem], f"Imagem {stem} do teste.", WIDTH * 0.8)
        block += [
            para(error.get("observation", error.get("description", "Observação não registrada."))),
            para(f"<b>Possível causa.</b> {escape(clean(error.get('hypothesis', 'Não registrada.')))}",
                 markup=True),
            para(f"<b>O que testar.</b> {escape(clean(error.get('proposed_correction', 'Não registrado.')))}",
                 markup=True),
        ]
        story.append(KeepTogether(block[:4]))
        story.extend(block[4:])

    # 8. Limitações e próximos passos
    heading("8. Limitações e próximos passos")
    story.append(ListFlowable(
        [ListItem(para(text), leftIndent=12) for text in source["limitations"]],
        bulletType="bullet", start="•", leftIndent=14, bulletFontName=styles["_head"],
        bulletFontSize=10, bulletOffsetY=0, bulletColor=MUTED))
    story.append(Spacer(1, 4))
    story.append(para(source["next_steps"]))

    # 9. Conclusão
    heading("9. Conclusão")
    story.append(para(source["conclusion"]))

    heading("Referências")
    for label, url in source["references"]:
        story.append(para(f'{escape(label)} Disponível em: <link href="{escape(url)}" color="#2C674C">'
                          f'{escape(url)}</link>.', "ref", markup=True))

    def first_page(canvas, document):
        pass

    def later_pages(canvas, document):
        canvas.saveState()
        canvas.setFont(styles["_head"], 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(A4[0] / 2, 1.3 * cm, str(document.page))
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.3 * cm, bottomMargin=2.2 * cm, title=source["title"],
        author="; ".join(str(m.get("name", "")) for m in members),
    )
    document.build(story, onFirstPage=first_page, onLaterPages=later_pages)
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
