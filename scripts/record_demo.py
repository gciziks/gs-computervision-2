from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_examples(manifest: list[dict], count: int = 2) -> list[dict]:
    rows = sorted(
        (row for row in manifest if row.get("split") == "test"),
        key=lambda row: row["image_id"],
    )
    selected = []
    for row in rows:
        candidate = (ROOT / "data/processed/taco" / row["prepared_file"]).resolve()
        candidate.relative_to((ROOT / "data/processed/taco").resolve())
        if candidate.is_file():
            if row.get("sha256") and sha256(candidate) != row["sha256"]:
                raise ValueError(f"Imagem não corresponde ao manifesto: {candidate.name}")
            selected.append({**row, "path": candidate})
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError("Prepare os dados: faltam imagens locais do teste para a gravação.")
    return selected


def make_montage(examples: list[dict], destination: Path) -> None:
    width, height, fps = 960, 540, 6
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError("O codec MP4 necessário à montagem não está disponível.")
    try:
        for row in examples:
            image = cv2.imdecode(np.fromfile(row["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Não foi possível ler {row['path'].name}.")
            scale = min(width / image.shape[1], (height - 60) / image.shape[0])
            resized = cv2.resize(image, None, fx=scale, fy=scale)
            frame = np.full((height, width, 3), 245, dtype=np.uint8)
            left = (width - resized.shape[1]) // 2
            top = (height - 60 - resized.shape[0]) // 2
            frame[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
            cv2.putText(frame, "Montagem do teste TACO - nao e captura real", (24, height - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 40, 35), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Imagem {row['image_id']} | 1,5 segundo por imagem",
                        (24, height - 10), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (50, 60, 55), 1, cv2.LINE_AA)
            for _ in range(9):
                writer.write(frame)
    finally:
        writer.release()
    converted = destination.with_name(destination.stem + "-h264.mp4")
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-i", str(destination), "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(converted)],
        check=True, capture_output=True,
    )
    converted.replace(destination)


def evidence_page(page, title: str, body: str) -> None:
    page.set_content(
        "<!doctype html><html lang='pt-BR'><meta charset='utf-8'>"
        "<style>body{margin:0;background:#f7f9f7;color:#19372b;font:24px/1.5 Arial,sans-serif}"
        "main{max-width:1050px;margin:90px auto;padding:0 48px}h1{font-size:48px;line-height:1.15}"
        "p,li{max-width:76ch}table{border-collapse:collapse;width:100%;margin:28px 0}"
        "th,td{text-align:left;padding:12px 16px;border-bottom:1px solid #cdd7d1}"
        "small{font-size:20px}code{font-size:20px}</style>"
        f"<main><h1>{html.escape(title)}</h1>{body}</main></html>"
    )


def hold(page, seconds: float) -> None:
    while seconds > 0:
        interval = min(seconds, 10)
        page.wait_for_timeout(interval * 1000)
        seconds -= interval


def record(args) -> Path:
    metrics_path = ROOT / "reports/metrics.json"
    manifest_path = ROOT / "data/manifests/taco.json"
    if not args.weights.is_file() or not metrics_path.is_file():
        raise ValueError("Conclua o treinamento e a avaliação antes de gravar a demonstração.")
    metrics = json.loads(metrics_path.read_text("utf-8"))
    if metrics["model"]["sha256"] != sha256(args.weights):
        raise ValueError("Os pesos diferem do checkpoint registrado nas métricas.")
    examples = test_examples(json.loads(manifest_path.read_text("utf-8")))
    evaluation = metrics["evaluation"]
    confidence = float(evaluation["confidence"])
    if not 0.05 <= confidence <= 0.95:
        raise ValueError("O limiar registrado não cabe no controle da interface.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Instale FFmpeg e deixe o executável disponível no PATH.")
    import threading
    import urllib.request

    import uvicorn
    from playwright.sync_api import sync_playwright

    from app.main import build_app

    workspace = ROOT / "tmp/recording" / uuid4().hex
    workspace.mkdir(parents=True)
    montage = workspace / "montagem-teste-nao-captura-real.mp4"
    make_montage(examples, montage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    screenshot = ROOT / "evidence/app-result.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    port = args.port or free_port()
    server = uvicorn.Server(uvicorn.Config(
        build_app(args.weights, args.device, args.imgsz),
        host="127.0.0.1", port=port, log_level="warning",
    ))
    server_thread = threading.Thread(target=server.run, daemon=True)
    browser = context = video = None
    app_started = False
    actions = []
    try:
        server_thread.start()
        app_started = True
        deadline = time.monotonic() + 30
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("A aplicação local não respondeu a tempo.")
                time.sleep(.2)
        with sync_playwright() as playwright:
            executable = args.browser_executable or Path(playwright.chromium.executable_path)
            if not executable.is_file():
                raise RuntimeError(
                    "Chromium não encontrado. Execute python -m playwright install chromium."
                )
            browser = playwright.chromium.launch(headless=True, executable_path=str(executable))
            context = browser.new_context(
                viewport={"width": 1280, "height": 900},
                record_video_dir=str(workspace / "browser"),
                record_video_size={"width": 1280, "height": 900},
                locale="pt-BR", color_scheme="light", reduced_motion="reduce",
            )
            page = context.new_page()
            video = page.video
            started = time.monotonic()
            evidence_page(page, "OlhoLimpo", (
                "<p>Detecção de resíduos em imagens e vídeos com treinamento customizado.</p>"
                "<p>Esta gravação executa a aplicação local, sem previsões simuladas. "
                "As imagens vêm do teste reservado do TACO.</p>"
                "<p>O trecho de vídeo é uma montagem de imagens estáticas, identificada como tal. "
                "Não há captura de webcam nem ensaio urbano real nesta gravação.</p>"
            ))
            hold(page, 9)
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.get_by_role("button", name="Analisar imagem", exact=True).wait_for()
            slider = page.get_by_role("slider")
            slider.focus()
            slider.press("Home")
            for _ in range(round((confidence - 0.05) / 0.05)):
                slider.press("ArrowRight")
            if abs(float(slider.input_value()) - confidence) > 0.001:
                raise RuntimeError("O controle de confiança difere do limiar avaliado.")
            hold(page, 5)
            previous_status = ""
            for index, row in enumerate(examples):
                page.locator("input[type=file][accept*='image']").first.set_input_files(
                    str(row["path"])
                )
                hold(page, 2)
                before = time.monotonic()
                page.get_by_role("button", name="Analisar imagem", exact=True).click()
                page.wait_for_function(
                    "previous => { const meta = document.querySelector('#image-meta'); "
                    "return !document.querySelector('#result-image').hidden && "
                    "meta.innerText.includes('Processamento') && meta.innerText !== previous; }",
                    arg=previous_status, timeout=args.timeout * 1000,
                )
                previous_status = page.locator("#image-meta").inner_text()
                if not previous_status or "Processamento" not in previous_status:
                    raise RuntimeError("A interface não apresentou um resultado real de imagem.")
                actions.append({"kind": "image", "image_id": row["image_id"],
                                "wait_seconds": time.monotonic() - before,
                                "status": previous_status})
                page.locator("#detections").scroll_into_view_if_needed()
                hold(page, 5)
                page.get_by_role("heading", name="OlhoLimpo", exact=True).scroll_into_view_if_needed()
                hold(page, 6)
                if index == 0:
                    page.screenshot(path=str(screenshot), full_page=True)
            page.get_by_role("tab", name="Vídeo", exact=True).click()
            page.locator("input[type=file][accept*='video']").first.set_input_files(str(montage))
            hold(page, 3)
            before = time.monotonic()
            page.get_by_role("button", name="Analisar vídeo", exact=True).click()
            page.locator("#result-video:not([hidden])").wait_for(timeout=args.timeout * 1000)
            actions.append({"kind": "video_montage", "source_image_ids": [
                row["image_id"] for row in examples], "wait_seconds": time.monotonic() - before,
                "source_is_real_capture": False})
            page.get_by_text("Resumo do processamento", exact=True).scroll_into_view_if_needed()
            hold(page, 9)
            page.get_by_role("heading", name="OlhoLimpo", exact=True).scroll_into_view_if_needed()
            page.locator("video").last.evaluate("video => video.play()")
            hold(page, 6)
            values = [
                ("Precisão", evaluation["precision"]),
                ("Recall", evaluation["recall"]),
                ("F1", evaluation["f1"]),
                ("AP50", evaluation["map50"]),
                ("AP50–95", evaluation["map50_95"]),
            ]
            table = "".join(f"<tr><td>{name}</td><td>{value:.3f}</td></tr>"
                            for name, value in values)
            evidence_page(page, "Resultados do teste reservado", (
                f"<table><tr><th>Métrica</th><th>Valor medido</th></tr>{table}</table>"
                f"<p>Limiar {confidence:.2f}, escolhido na validação. "
                f"Teste: {int(metrics['dataset']['test_images'])} imagens. "
                "Fonte: reports/metrics.json.</p>"
                "<p><small>Esta página resume as métricas registradas; não substitui a "
                "execução da aplicação mostrada anteriormente.</small></p>"
            ))
            hold(page, 14)
            limitations = "".join(f"<li>{html.escape(str(item))}</li>"
                                  for item in metrics.get("limitations", []))
            evidence_page(page, "Limites e reprodução", (
                f"<ul>{limitations}</ul>"
                "<p>O modelo identifica aparência visual, sem comprovar descarte irregular "
                "ou reciclabilidade. O vídeo demonstrado é uma montagem de imagens do teste.</p>"
                "<p>O README descreve instalação, preparação dos dados, treino, avaliação "
                "e execução local. Os resultados anotados podem ser exportados em imagem, "
                "vídeo, JSON e CSV.</p>"
            ))
            hold(page, max(10, args.minimum_seconds - (time.monotonic() - started)))
            context.close()
            context = None
            recording = Path(video.path())
            browser.close()
            browser = None
        subprocess.run(
            [ffmpeg, "-y", "-i", str(recording), "-an", "-vf", "fps=15,format=yuv420p",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-movflags", "+faststart",
             str(args.output)],
            check=True, capture_output=True, text=True,
        )
        audit = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "weights_sha256": metrics["model"]["sha256"],
            "metrics_sha256": sha256(metrics_path), "confidence": confidence,
            "actions": actions, "camera_used": False, "predictions_mocked": False,
            "video_source": "montagem identificada de imagens estáticas do teste TACO",
            "output": args.output.name,
        }
        (ROOT / "evidence/demo-recording.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), "utf-8"
        )
        return args.output
    finally:
        if context is not None:
            with suppress(Exception):
                context.close()
        if browser is not None:
            with suppress(Exception):
                browser.close()
        if app_started:
            server.should_exit = True
            server_thread.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description="Grava a aplicação local com inferência real.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "deliverables/olholimpo-demo.mp4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--minimum-seconds", type=int, default=90)
    parser.add_argument("--browser-executable", type=Path)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or args.timeout < 1 or args.minimum_seconds < 30:
        parser.error("Use porta válida, timeout positivo e duração mínima de 30 segundos.")
    print(record(args))


if __name__ == "__main__":
    main()
