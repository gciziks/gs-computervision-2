from __future__ import annotations

import argparse
import logging
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
from olholimpo.inference import inference_defaults
from app.presentation import build_presentation

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
FAVICON = ROOT / "assets" / "icon.svg"

# Identidade "caderno de vistoria": papel frio quadriculado, tinta verde-abeto e
# âmbar de sinalização reservado às molduras de visor que enquadram as imagens.
PALETTE = {
    "paper": "#eef1ec", "sheet": "#fbfcfa", "ink": "#16291f", "muted": "#56685d",
    "rule": "#cbd4cb", "moss": "#2c674c", "moss_deep": "#1d4a36", "signal": "#d99a12",
    "paper_dark": "#121c17", "sheet_dark": "#19261f", "ink_dark": "#e3eae4",
    "muted_dark": "#9db0a4", "rule_dark": "#2d3f35", "moss_dark": "#62b08a", "signal_dark": "#f0b93d",
}

HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=JetBrains+Mono:wght@400;600&display=swap">'
)

HEADER = """
<header class="ol-header">
  <div class="ol-frame">
    <span class="ol-tag" aria-hidden="true">resíduo 0,35+</span>
    <h1>OlhoLimpo</h1>
  </div>
  <p>Detector de resíduos em imagens e vídeos. Envie um arquivo, ajuste a confiança
  mínima e confira cada região apontada pelo modelo treinado com o TACO.</p>
</header>
"""

CSS = """
:root {
  --ol-paper: %(paper)s; --ol-sheet: %(sheet)s; --ol-ink: %(ink)s; --ol-muted: %(muted)s;
  --ol-rule: %(rule)s; --ol-moss: %(moss)s; --ol-signal: %(signal)s;
}
.dark {
  --ol-paper: %(paper_dark)s; --ol-sheet: %(sheet_dark)s; --ol-ink: %(ink_dark)s;
  --ol-muted: %(muted_dark)s; --ol-rule: %(rule_dark)s; --ol-moss: %(moss_dark)s;
  --ol-signal: %(signal_dark)s;
}
body, gradio-app {
  background-color: var(--ol-paper) !important;
  background-image:
    linear-gradient(color-mix(in srgb, var(--ol-ink) 5%%, transparent) 1px, transparent 1px),
    linear-gradient(90deg, color-mix(in srgb, var(--ol-ink) 5%%, transparent) 1px, transparent 1px) !important;
  background-size: 28px 28px !important;
}
.gradio-container { max-width: 1280px !important; font-variant-numeric: tabular-nums; }
::selection { background: color-mix(in srgb, var(--ol-signal) 40%%, transparent); color: var(--ol-ink); }
:is(button, input, select, textarea, a, [tabindex]):focus-visible {
  outline: 2px solid var(--ol-signal) !important; outline-offset: 3px;
}

/* Cabeçalho: a marca é enquadrada como uma detecção do próprio modelo. */
.ol-header { display: grid; gap: 1.1rem; padding: 2.25rem 0 1.5rem; }
.ol-frame { position: relative; justify-self: start; padding: .5rem 1.1rem .35rem; }
.ol-frame::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  --c: var(--ol-signal); --l: 18px; --w: 3px;
  background-image: linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c));
  background-position: 0 0, 0 0, 100%% 0, 100%% 0, 0 100%%, 0 100%%, 100%% 100%%, 100%% 100%%;
  background-size: var(--l) var(--w), var(--w) var(--l), var(--l) var(--w), var(--w) var(--l), var(--l) var(--w), var(--w) var(--l), var(--l) var(--w), var(--w) var(--l);
  background-repeat: no-repeat;
}
.ol-tag {
  position: absolute; left: 0; bottom: 100%%; padding: .1rem .45rem;
  background: var(--ol-signal); color: #1b1404; font-size: .72rem; font-weight: 600;
  letter-spacing: .01em; line-height: 1.4;
}
.ol-header h1 {
  margin: 0 !important; color: var(--ol-ink) !important;
  font-size: clamp(2.6rem, 6vw, 4.4rem) !important; font-weight: 800 !important;
  font-stretch: 125%%; font-variation-settings: "wdth" 125; letter-spacing: -.02em; line-height: 1;
}
.ol-header p { margin: 0; max-width: 62ch; color: var(--ol-muted); font-size: 1.05rem; line-height: 1.55; }

/* Faixa de controle: estado do modelo e limiar lado a lado. */
#inference-controls {
  background: var(--ol-sheet); border: 1px solid var(--ol-rule); border-left: 4px solid var(--ol-moss);
  border-radius: 6px; padding: .85rem 1.1rem; gap: .35rem;
}
#inference-controls > .block { border: 0 !important; box-shadow: none !important; padding: 0 !important; background: transparent !important; }
#model-status, #model-status p { font-size: .9rem; color: var(--ol-muted); margin: 0; }
#model-status strong { color: var(--ol-ink); }

/* Abas como fichas de um arquivo. */
.tab-wrapper, .tab-container { border-bottom-color: var(--ol-rule) !important; }
button[role="tab"] { font-weight: 600 !important; font-size: 1rem !important; padding-inline: 1.1rem !important; }
button[role="tab"][aria-selected="true"] { color: var(--ol-ink) !important; }

/* Molduras de visor nas imagens de entrada e resultado. */
.viewfinder { position: relative; overflow: visible !important; }
.viewfinder::after {
  content: ""; position: absolute; inset: -5px; pointer-events: none; z-index: 2;
  --c: var(--ol-rule); --l: 22px; --w: 2px;
  background-image: linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c)), linear-gradient(var(--c), var(--c));
  background-position: 0 0, 0 0, 100%% 0, 100%% 0, 0 100%%, 0 100%%, 100%% 100%%, 100%% 100%%;
  background-size: var(--l) var(--w), var(--w) var(--l), var(--l) var(--w), var(--w) var(--l), var(--l) var(--w), var(--w) var(--l), var(--l) var(--w), var(--w) var(--l);
  background-repeat: no-repeat;
}
.viewfinder.result::after { --c: var(--ol-signal); }

#image-status, #image-status p { font-size: .95rem; line-height: 1.55; }
#image-status strong { color: var(--ol-moss); }
.table-wrap, table, .cell-wrap { font-family: inherit !important; }
.ol-file .empty { height: 4.5rem !important; min-height: 0 !important; }
.ol-file .empty .icon { height: 1.4rem; opacity: .5; }

#methodology { max-width: 80ch; margin: 2rem auto 0; }
#presentation p, #presentation ul { max-width: 78ch; line-height: 1.6; }
#presentation h2 { font-stretch: 115%%; font-variation-settings: "wdth" 115; font-weight: 800; letter-spacing: -.01em; }
#presentation h3 { margin-top: 1.75rem; margin-bottom: .6rem; font-weight: 700; }
#presentation table { font-variant-numeric: tabular-nums; border-collapse: collapse; }
#presentation :is(table, tr) { border: 0 !important; box-shadow: none !important; }
#presentation :is(th, td) { border: 0 !important; border-bottom: 1px solid var(--ol-rule) !important; padding: .55rem 1.25rem .55rem 0 !important; background: transparent !important; }
#presentation th { color: var(--ol-muted); font-weight: 600; border-bottom: 2px solid var(--ol-ink) !important; }
#presentation a { color: var(--ol-moss); text-underline-offset: .18em; }

@media (max-width: 640px) {
  .ol-header { padding-top: 1.75rem; }
  #inference-controls { padding: .75rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
""" % PALETTE


def build_theme():
    import gradio as gr

    p = PALETTE
    font = ["Archivo", "ui-sans-serif", "system-ui", "sans-serif"]
    mono = ["JetBrains Mono", "ui-monospace", "Consolas", "monospace"]
    return gr.themes.Base(
        primary_hue="emerald", neutral_hue="stone", font=font, font_mono=mono,
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill=p["paper"], body_background_fill_dark=p["paper_dark"],
        body_text_color=p["ink"], body_text_color_dark=p["ink_dark"],
        body_text_color_subdued=p["muted"], body_text_color_subdued_dark=p["muted_dark"],
        background_fill_primary=p["sheet"], background_fill_primary_dark=p["sheet_dark"],
        background_fill_secondary=p["paper"], background_fill_secondary_dark=p["paper_dark"],
        block_background_fill=p["sheet"], block_background_fill_dark=p["sheet_dark"],
        block_border_color=p["rule"], block_border_color_dark=p["rule_dark"],
        border_color_primary=p["rule"], border_color_primary_dark=p["rule_dark"],
        block_shadow="none", block_shadow_dark="none",
        block_label_background_fill=p["sheet"], block_label_background_fill_dark=p["sheet_dark"],
        block_label_text_color=p["muted"], block_label_text_color_dark=p["muted_dark"],
        block_label_border_color=p["rule"], block_label_border_color_dark=p["rule_dark"],
        block_title_text_weight="600", block_label_text_weight="600",
        color_accent=p["moss"], color_accent_soft="#dde9e1", color_accent_soft_dark="#21382c",
        border_color_accent=p["moss"], border_color_accent_dark=p["moss_dark"],
        slider_color=p["moss"], slider_color_dark=p["moss_dark"],
        loader_color=p["signal"], loader_color_dark=p["signal_dark"],
        link_text_color=p["moss"], link_text_color_dark=p["moss_dark"],
        input_background_fill=p["sheet"], input_background_fill_dark=p["sheet_dark"],
        input_border_color=p["rule"], input_border_color_dark=p["rule_dark"],
        input_border_color_focus=p["moss"], input_border_color_focus_dark=p["moss_dark"],
        table_border_color=p["rule"], table_border_color_dark=p["rule_dark"],
        table_even_background_fill=p["sheet"], table_even_background_fill_dark=p["sheet_dark"],
        table_odd_background_fill="#f3f6f2", table_odd_background_fill_dark="#1d2b23",
        code_background_fill="#e4e9e3", code_background_fill_dark="#22332a",
        button_primary_background_fill=p["moss"], button_primary_background_fill_dark=p["moss_dark"],
        button_primary_background_fill_hover=p["moss_deep"],
        button_primary_background_fill_hover_dark="#7cc3a0",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#0e1a14",
        button_primary_border_color=p["moss"], button_primary_border_color_dark=p["moss_dark"],
        button_secondary_background_fill=p["sheet"], button_secondary_background_fill_dark=p["sheet_dark"],
        button_secondary_border_color=p["rule"], button_secondary_border_color_dark=p["rule_dark"],
        button_large_text_weight="700", button_large_radius="6px",
        button_transform_active="translateY(1px)",
    )


def _rows(detections: list[dict]) -> list[list]:
    return [
        [
            item.get("label", "resíduo").replace("residuo", "resíduo"),
            round(float(item["confidence"]), 3),
            *[round(float(value), 1) for value in item["bbox_xyxy"]],
        ]
        for item in detections
    ]


def _duration(timings: dict) -> str:
    for key in ("total", "total_ms", "end_to_end"):
        if key in timings:
            return f"{float(timings[key]):.1f} ms"
    values = [float(value) for value in timings.values() if isinstance(value, (int, float))]
    return f"{sum(values):.1f} ms" if values else "não informado"


def _failure(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "**Modelo ou arquivo não encontrado.** Confira o caminho dos pesos e selecione o arquivo novamente."
    if isinstance(error, (ValueError, OSError)):
        return f"**Não foi possível analisar o arquivo.** {error}"
    LOGGER.exception("Falha no processamento local")
    return "**O processamento falhou.** Confira o erro no terminal, os pesos e a memória disponível antes de tentar novamente."


class DemoController:
    def __init__(
        self,
        weights: str | Path = ROOT / "models" / "best.pt",
        device: str = "cpu",
        imgsz: int | None = None,
        output_dir: str | Path = ROOT / "outputs" / "demo",
    ):
        self.weights = Path(weights).resolve()
        self.device = device
        defaults = inference_defaults(self.weights)
        self.imgsz = imgsz or defaults["imgsz"]
        self.confidence = defaults["confidence"]
        self.output_dir = Path(output_dir).resolve()
        self._detector = None
        self._lock = Lock()

    def detector(self):
        with self._lock:
            if self._detector is None:
                from olholimpo.inference import Detector

                self._detector = Detector(
                    weights=str(self.weights), device=self.device, imgsz=self.imgsz
                )
            return self._detector

    def model_status(self) -> str:
        if self._detector is not None:
            info = self._detector.model_info
            return (
                f"Modelo **{Path(info.get('path', self.weights)).name}** carregado em "
                f"**{info.get('device', self.device)}**, SHA-256 "
                f"`{str(info.get('sha256', 'não informado'))[:12]}`"
            )
        if not self.weights.is_file():
            return (
                f"**Pesos ausentes: {self.weights.name}.** Consulte o README para obter o "
                "checkpoint customizado. Nenhum modelo genérico será usado em seu lugar."
            )
        return (
            f"Modelo **{self.weights.name}** pronto para carregar em **{self.device}** "
            "na primeira análise."
        )

    def _destination(self) -> Path:
        destination = self.output_dir / uuid4().hex
        destination.mkdir(parents=True, exist_ok=False)
        return destination

    def analyze_image(self, image: str | np.ndarray | None, confidence: float):
        empty = (None, [], None, None, None)
        if image is None:
            return (*empty, "Selecione uma imagem antes de analisar.", self.model_status())
        try:
            if isinstance(image, str):
                from olholimpo.inference import read_image

                pixels = read_image(image)
            else:
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError("Use uma imagem RGB de três canais.")
                if image.shape[0] * image.shape[1] > 40_000_000:
                    raise ValueError("A imagem ultrapassa 40 milhões de pixels. Reduza a resolução.")
                pixels = np.ascontiguousarray(image[:, :, ::-1])
            from olholimpo.inference import save_result

            result = self.detector().predict(
                pixels, confidence=float(confidence)
            )
            files = save_result(result, self._destination())
            count = len(result.detections)
            count_text = "1 resíduo detectado" if count == 1 else f"{count} resíduos detectados"
            status = (
                f"**{count_text}** acima de {float(confidence):.2f}. "
                f"Processamento: **{_duration(result.timings_ms)}**. "
                "As coordenadas estão em pixels da imagem original."
            )
            if not count:
                status += " Nenhuma detecção não garante que o ambiente esteja sem resíduos."
            return (
                np.ascontiguousarray(result.annotated[:, :, ::-1]),
                _rows(result.detections),
                str(files["image"]),
                str(files["json"]),
                str(files["csv"]),
                status,
                self.model_status(),
            )
        except Exception as error:
            return (*empty, _failure(error), self.model_status())

    def analyze_video(self, video: str | None, confidence: float):
        empty = (None, None, None, None)
        if not video:
            return (*empty, {"estado": "Selecione um vídeo antes de analisar."}, self.model_status())
        try:
            path = Path(video)
            if not path.is_file():
                raise ValueError("O arquivo de vídeo não está disponível. Selecione-o novamente.")
            if path.stat().st_size > 100 * 1024 * 1024:
                raise ValueError("O vídeo ultrapassa 100 MB. Envie um trecho menor.")
            from olholimpo.inference import process_video

            files = process_video(
                str(path),
                self._destination(),
                self.detector(),
                confidence=float(confidence),
                max_seconds=30,
            )
            return (
                str(files["video"]),
                str(files["video"]),
                str(files["json"]),
                str(files["csv"]),
                files["summary"],
                self.model_status(),
            )
        except Exception as error:
            return (*empty, {"erro": _failure(error)}, self.model_status())


def build_app(
    weights: str | Path = ROOT / "models" / "best.pt",
    device: str = "cpu",
    imgsz: int | None = None,
):
    import gradio as gr

    controller = DemoController(weights=weights, device=device, imgsz=imgsz)
    headers = ["Objeto", "Confiança", "x1", "y1", "x2", "y2"]
    with gr.Blocks(title="OlhoLimpo", analytics_enabled=False) as demo:
        gr.HTML(HEADER, elem_id="intro")
        with gr.Column(elem_id="inference-controls") as inference_controls:
            model_status = gr.Markdown(controller.model_status(), elem_id="model-status")
            confidence = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=controller.confidence,
                step=0.05,
                label="Confiança mínima",
                info="Valores menores mostram mais candidatos e podem aumentar os falsos positivos.",
            )
        with gr.Tab("Imagem") as image_tab:
            with gr.Row(equal_height=True):
                with gr.Column():
                    image_input = gr.Image(
                        label="Imagem de entrada (até 20 MB)",
                        type="filepath", image_mode="RGB", sources=["upload"],
                        elem_classes=["viewfinder"],
                    )
                    image_button = gr.Button("Analisar imagem", variant="primary")
                    samples = sorted(
                        path for path in (ROOT / "data" / "demo").glob("*")
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    )
                    if samples:
                        gr.Examples(
                            examples=[[str(path)] for path in samples[:8]],
                            inputs=[image_input],
                            label="Amostras de demonstração",
                            cache_examples=False,
                        )
                with gr.Column():
                    image_output = gr.Image(
                        label="Resultado", type="numpy", interactive=False,
                        elem_classes=["viewfinder", "result"],
                    )
                    image_status = gr.Markdown(
                        "As regiões detectadas aparecem aqui depois da análise.", elem_id="image-status"
                    )
            detections = gr.Dataframe(
                headers=headers,
                datatype=["str", "number", "number", "number", "number", "number"],
                value=[],
                label="Detecções",
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                image_file = gr.File(label="Imagem anotada", interactive=False, elem_classes=["ol-file"])
                image_json = gr.File(label="Detecções em JSON", interactive=False, elem_classes=["ol-file"])
                image_csv = gr.File(label="Detecções em CSV", interactive=False, elem_classes=["ol-file"])
            image_button.click(
                controller.analyze_image,
                inputs=[image_input, confidence],
                outputs=[
                    image_output, detections, image_file, image_json, image_csv,
                    image_status, model_status,
                ],
                concurrency_limit=1,
                concurrency_id="detector",
                show_progress="full",
                api_visibility="private",
            )
        with gr.Tab("Vídeo") as video_tab:
            gr.Markdown(
                "Use um trecho de até 30 segundos e 100 MB. "
                "Prefira MP4 com vídeo H.264 para pré-visualização no navegador. "
                "O arquivo é processado antes da reprodução; o resumo informa a velocidade medida."
            )
            with gr.Row():
                video_input = gr.Video(
                    label="Vídeo de entrada", sources=["upload"], elem_classes=["viewfinder"]
                )
                video_output = gr.Video(
                    label="Vídeo anotado", interactive=False, elem_classes=["viewfinder", "result"]
                )
            video_button = gr.Button("Analisar vídeo", variant="primary")
            video_summary = gr.JSON(label="Resumo do processamento")
            with gr.Row():
                video_file = gr.File(label="Vídeo anotado", interactive=False, elem_classes=["ol-file"])
                video_json = gr.File(label="Detecções em JSON", interactive=False, elem_classes=["ol-file"])
                video_csv = gr.File(label="Detecções em CSV", interactive=False, elem_classes=["ol-file"])
            video_button.click(
                controller.analyze_video,
                inputs=[video_input, confidence],
                outputs=[
                    video_output, video_file, video_json, video_csv, video_summary, model_status,
                ],
                concurrency_limit=1,
                concurrency_id="detector",
                show_progress="full",
                api_visibility="private",
            )
        with gr.Tab("Apresentação") as presentation_tab:
            build_presentation(ROOT)
        with gr.Accordion("Método e limites", open=False, elem_id="methodology") as methodology:
            gr.Markdown(
                "O detector usa uma classe agregada, **resíduo**, ajustada com anotações do TACO. "
                "A confiança é um escore do modelo, não uma probabilidade calibrada. "
                "Caixas em quadros diferentes podem representar o mesmo objeto.\n\n"
                "O modelo identifica aparência visual: não comprova descarte irregular, "
                "reciclabilidade ou risco sanitário. Objetos pequenos, sobrepostos e fundos "
                "carregados podem causar erros. Confira as métricas e os exemplos de falha no relatório.\n\n"
                "A captura contínua da webcam é executada pelo comando descrito no README. "
                "Use embalagens limpas em cenário controlado e identifique esse ensaio na apresentação.\n\n"
                "Os arquivos exportados ficam em `outputs/demo` nesta máquina. "
                "Apague-os após o uso se contiverem imagens pessoais. "
                "A aplicação não cria um link público."
            )
        for tab in (image_tab, video_tab):
            tab.select(lambda: (gr.update(visible=True), gr.update(visible=True)),
                       outputs=[inference_controls, methodology], api_visibility="private", queue=False)
        presentation_tab.select(lambda: (gr.update(visible=False), gr.update(visible=False)),
                                outputs=[inference_controls, methodology], api_visibility="private", queue=False)
    return demo


def main():
    parser = argparse.ArgumentParser(description="Demonstração local do detector OlhoLimpo.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--device", default="cpu", help="cpu ou índice CUDA, por exemplo 0")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    import gradio as gr

    demo = build_app(args.weights, args.device, args.imgsz)
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        footer_links=[],
        max_file_size="100mb",
        theme=build_theme(),
        css=CSS,
        head=HEAD,
        favicon_path=str(FAVICON),
        allowed_paths=[str(ROOT / "outputs" / "demo")],
    )


if __name__ == "__main__":
    main()
