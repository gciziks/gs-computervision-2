import argparse
import json
import time
from pathlib import Path

import cv2

from olholimpo.inference import Detector, inference_defaults, process_video, read_image, save_result


def main():
    parser = argparse.ArgumentParser(description="Detectar resíduos em imagem, vídeo ou webcam.")
    parser.add_argument("source", help="Arquivo local ou 'webcam'.")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    try:
        defaults = inference_defaults(args.weights)
        args.imgsz = args.imgsz or defaults["imgsz"]
        if args.confidence is None:
            args.confidence = defaults["confidence"]
        detector = Detector(args.weights, args.device, args.imgsz)
        if args.source == "webcam":
            camera = cv2.VideoCapture(args.camera)
            try:
                if not camera.isOpened():
                    raise ValueError("Câmera indisponível. Verifique a permissão ou use um vídeo.")
                while True:
                    started = time.perf_counter()
                    ok, frame = camera.read()
                    if not ok:
                        raise ValueError("Não foi possível ler a câmera.")
                    result = detector.predict(frame, args.confidence)
                    fps = 1 / max(time.perf_counter() - started, 1e-6)
                    cv2.putText(result.annotated, f"{fps:.1f} FPS | Q: sair | S: salvar",
                                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                    cv2.imshow("OlhoLimpo", result.annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("s"):
                        print(json.dumps(save_result(result, args.output), ensure_ascii=False))
            finally:
                camera.release()
                cv2.destroyAllWindows()
        elif Path(args.source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            print(json.dumps(process_video(args.source, args.output, detector, args.confidence),
                             ensure_ascii=False, indent=2))
        else:
            result = detector.predict(read_image(args.source), args.confidence)
            print(json.dumps(save_result(result, args.output), ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Erro: {error}\n")


if __name__ == "__main__":
    main()
