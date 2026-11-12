from __future__ import annotations

import argparse
import logging
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
from olholimpo.inference import inference_defaults

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CSS = """
.gradio-container { max-width: 1320px !important; }
#intro { max-width: 74ch; margin-bottom: 1.25rem; }
#model-status { font-size: .9rem; }
#methodology { max-width: 80ch; }
button:focus-visible, input:focus-visible { outline: 2px solid #17644d; outline-offset: 3px; }
::selection { background: #c6e6d9; color: #12392c; }
"""


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
                f"Modelo: **{Path(info.get('path', self.weights)).name}** · "
                f"dispositivo: **{info.get('device', self.device)}** · "
                f"SHA-256: `{str(info.get('sha256', 'não informado'))[:12]}`"
            )
        if not self.weights.is_file():
            return (
                f"**Pesos ausentes: {self.weights.name}.** Consulte o README para obter o "
                "checkpoint customizado. Nenhum modelo genérico será usado em seu lugar."
            )
        return (
            f"Modelo: **{self.weights.name}** · dispositivo solicitado: **{self.device}**. "
            "Carregamento na primeira análise."
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
        gr.Markdown(
            "# OlhoLimpo\n"
            "Detecção de resíduos em imagens e vídeos. Envie um arquivo, ajuste a confiança "
            "e confira as regiões apontadas pelo modelo customizado.",
            elem_id="intro",
        )
        model_status = gr.Markdown(controller.model_status(), elem_id="model-status")
        confidence = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=controller.confidence,
            step=0.05,
            label="Confiança mínima",
            info="Valores menores mostram mais candidatos e podem aumentar os falsos positivos.",
        )
        with gr.Tab("Imagem"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    image_input = gr.Image(
                        label="Imagem de entrada (até 20 MB)",
                        type="filepath", image_mode="RGB", sources=["upload"],
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
                    image_output = gr.Image(label="Resultado", type="numpy", interactive=False)
                    image_status = gr.Markdown("O resultado aparecerá após a análise.")
            detections = gr.Dataframe(
                headers=headers,
                datatype=["str", "number", "number", "number", "number", "number"],
                value=[],
                label="Detecções",
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                image_file = gr.File(label="Imagem anotada", interactive=False)
                image_json = gr.File(label="Detecções em JSON", interactive=False)
                image_csv = gr.File(label="Detecções em CSV", interactive=False)
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
        with gr.Tab("Vídeo"):
            gr.Markdown(
                "Use um trecho de até 30 segundos e 100 MB. "
                "Prefira MP4 com vídeo H.264 para pré-visualização no navegador. "
                "O arquivo é processado antes da reprodução; o resumo informa a velocidade medida."
            )
            with gr.Row():
                video_input = gr.Video(label="Vídeo de entrada", sources=["upload"])
                video_output = gr.Video(label="Vídeo anotado", interactive=False)
            video_button = gr.Button("Analisar vídeo", variant="primary")
            video_summary = gr.JSON(label="Resumo do processamento")
            with gr.Row():
                video_file = gr.File(label="Vídeo anotado", interactive=False)
                video_json = gr.File(label="Detecções em JSON", interactive=False)
                video_csv = gr.File(label="Detecções em CSV", interactive=False)
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
        with gr.Accordion("Método e limites", open=False, elem_id="methodology"):
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
        theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"),
        css=CSS,
        allowed_paths=[str(ROOT / "outputs" / "demo")],
    )


if __name__ == "__main__":
    main()
