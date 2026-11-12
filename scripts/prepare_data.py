from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import time
from urllib.parse import urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, __version__ as pillow_version
from olholimpo.data import (
    assign_splits,
    coco_bbox_to_yolo,
    difference_hash,
    group_similar,
    normalize_annotation_image,
    safe_relative_path,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/3844bcd3535cd12ec0218494d1d79f1372f553cc/data/annotations.json"
ANNOTATIONS_SHA256 = "812c2e3f0d336b437f276597715de2abb08bb8a641599c45f593d233ec66457b"
ALLOWED_HOSTS = {
    "img.tacodataset.org",
    "olm-s3.s3.eu-west-1.amazonaws.com",
    "farm1.staticflickr.com",
    "farm2.staticflickr.com",
    "live.staticflickr.com",
}


def fetch(url: str, attempts: int = 3) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL de origem inválida")
    last_error = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "OlhoLimpo-academic-dataset/1.0"})
            with urlopen(request, timeout=60) as response:
                return response.read(40 * 1024 * 1024 + 1)
        except Exception as error:
            last_error = error
            if isinstance(error, HTTPError) and error.code in {400, 401, 403, 404, 410}:
                break
            if attempt + 1 < attempts:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Falha no download: {last_error}")


def download_image(image: dict, raw_dir: Path, max_side: int) -> dict:
    relative = safe_relative_path(image["file_name"])
    target = raw_dir / relative
    metadata_path = target.with_suffix(".meta.json")
    if target.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("sha256") == sha256_file(target)
            and metadata.get("max_side") == max_side
            and metadata.get("preprocessing_version") == 2
        ):
            with Image.open(target) as opened:
                opened.verify()
            return metadata
    url = image.get("flickr_url") or image.get("coco_url")
    if not url:
        raise ValueError("Imagem sem URL de origem")
    host = urlparse(url).hostname
    if host not in ALLOWED_HOSTS and not (host or "").endswith(".staticflickr.com"):
        raise ValueError(f"Host fora da distribuição TACO: {host}")
    download_url = url
    try:
        content = fetch(url)
    except RuntimeError:
        thumbnail_url = image.get("flickr_640_url")
        if not thumbnail_url or urlparse(thumbnail_url).hostname not in ALLOWED_HOSTS:
            raise
        download_url = thumbnail_url
        content = fetch(download_url)
    if len(content) > 40 * 1024 * 1024:
        raise ValueError("Imagem excede 40 MB")
    with Image.open(io.BytesIO(content)) as opened:
        opened.load()
        orientation = opened.getexif().get(274, 1)
        opened = normalize_annotation_image(
            opened, image["width"], image["height"], allow_resized=download_url != url
        )
        source_size = opened.size
        dhash = difference_hash(opened)
        converted = opened.convert("RGB")
        converted.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part")
        converted.save(temporary, format="JPEG", quality=92)
        temporary.replace(target)
        stored_size = converted.size
    metadata = {
        "image_id": image["id"],
        "file_name": relative.as_posix(),
        "source_url": url,
        "download_url": download_url,
        "license_recorded": image.get("license"),
        "license_status": "unspecified_cc"
        if image.get("license") == "CC"
        else "recorded_odbl"
        if str(image.get("license", "")).startswith("ODBL")
        else "unrecorded",
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "sha256": sha256_file(target),
        "dhash": f"{dhash:016x}",
        "original_width": image["width"],
        "original_height": image["height"],
        "stored_width": stored_size[0],
        "stored_height": stored_size[1],
        "max_side": max_side,
        "download_width": source_size[0],
        "download_height": source_size[1],
        "exif_orientation": orientation,
        "preprocessing_version": 2,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara a parte oficial do TACO para detecção de resíduos."
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/taco")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/taco")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-distance", type=int, default=6)
    parser.add_argument(
        "--include-unrecorded-license",
        action="store_true",
        help="Inclui para pesquisa imagens sem licença registrada; não autoriza redistribuição.",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 320 <= args.max_side <= 4096:
        parser.error("Use 1–16 workers e max-side entre 320 e 4096")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = args.raw_dir / "annotations.json"
    if not annotation_path.is_file():
        annotation_path.write_bytes(fetch(ANNOTATIONS_URL))
    if sha256_file(annotation_path) != ANNOTATIONS_SHA256:
        raise ValueError(
            "O hash das anotações não corresponde à versão oficial fixada neste preparo"
        )
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in annotations["categories"]}
    by_image = defaultdict(list)
    for annotation in annotations["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    eligible = []
    excluded = []
    for image in annotations["images"]:
        if not args.include_unrecorded_license and not image.get("license"):
            excluded.append({"image_id": image["id"], "reason": "license_not_recorded"})
            continue
        try:
            boxes = by_image[image["id"]]
            if not boxes:
                raise ValueError("Sem caixas revisadas; não assumir negativo")
            for annotation in boxes:
                if annotation["category_id"] not in categories or annotation.get("iscrowd", 0):
                    raise ValueError("Categoria desconhecida ou anotação crowd")
                coco_bbox_to_yolo(annotation["bbox"], image["width"], image["height"])
            eligible.append(image)
        except ValueError as error:
            excluded.append({"image_id": image["id"], "reason": str(error)})
    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(download_image, image, args.raw_dir, args.max_side): image
            for image in eligible
        }
        for count, future in enumerate(as_completed(pending), 1):
            image = pending[future]
            try:
                record = future.result()
                record["annotation_count"] = len(by_image[image["id"]])
                record["source_category_counts"] = dict(
                    Counter(categories[box["category_id"]] for box in by_image[image["id"]])
                )
                records.append(record)
            except Exception as error:
                failures.append(
                    {"image_id": image["id"], "file_name": image["file_name"], "error": str(error)}
                )
            if count % 25 == 0 or count == len(eligible):
                print(
                    f"Imagens verificadas: {count}/{len(eligible)}; falhas: {len(failures)}",
                    flush=True,
                )
    seen = {}
    unique = []
    for record in sorted(records, key=lambda item: item["image_id"]):
        if record["sha256"] in seen:
            excluded.append(
                {
                    "image_id": record["image_id"],
                    "reason": "exact_duplicate",
                    "duplicate_of": seen[record["sha256"]],
                }
            )
        else:
            seen[record["sha256"]] = record["image_id"]
            unique.append(record)
    groups = group_similar(unique, distance=args.hash_distance)
    manifest = assign_splits(groups, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifests.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        expected_stems = {
            f"{record['image_id']:06d}" for record in manifest if record["split"] == split
        }
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
            directory = args.output / kind / split
            if directory.is_dir():
                for existing in directory.iterdir():
                    if (
                        existing.is_file()
                        and existing.suffix == suffix
                        and re.fullmatch(r"\d{6}", existing.stem)
                        and existing.stem not in expected_stems
                    ):
                        existing.unlink()
    for record in manifest:
        name = f"{record['image_id']:06d}.jpg"
        record["prepared_file"] = f"images/{record['split']}/{name}"
        image_path = args.output / record["prepared_file"]
        label_path = args.output / "labels" / record["split"] / Path(name).with_suffix(".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = args.raw_dir / safe_relative_path(record["file_name"])
        if not image_path.exists():
            try:
                os.link(source_path, image_path)
            except OSError:
                shutil.copyfile(source_path, image_path)
        elif sha256_file(image_path) != record["sha256"]:
            shutil.copyfile(source_path, image_path)
        lines = []
        for box in by_image[record["image_id"]]:
            normalized = coco_bbox_to_yolo(
                box["bbox"], record["original_width"], record["original_height"]
            )
            lines.append("0 " + " ".join(f"{value:.8f}" for value in normalized))
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        record["label_sha256"] = sha256_file(label_path)
    for split in ("train", "val", "test"):
        paths = [record["prepared_file"] for record in manifest if record["split"] == split]
        (args.output / f"{split}.txt").write_text(
            "\n".join("./" + path for path in paths) + "\n", encoding="utf-8"
        )
    yaml = f"path: {json.dumps(args.output.resolve().as_posix())}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: residuo\n"
    (args.output / "data.yaml").write_text(yaml, encoding="utf-8")
    summary = {
        "preprocessing_version": 2,
        "python": platform.python_version(),
        "pillow": pillow_version,
        "source": ANNOTATIONS_URL,
        "annotations_sha256": sha256_file(annotation_path),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "hash_distance": args.hash_distance,
        "source_images": len(annotations["images"]),
        "source_annotations": len(annotations["annotations"]),
        "eligible_images": len(eligible),
        "prepared_images": len(manifest),
        "prepared_annotations": sum(item["annotation_count"] for item in manifest),
        "split_counts": dict(Counter(item["split"] for item in manifest)),
        "split_annotation_counts": {
            split: sum(item["annotation_count"] for item in manifest if item["split"] == split)
            for split in ("train", "val", "test")
        },
        "groups": len(groups),
        "largest_group": max(map(len, groups)),
        "license_counts": dict(Counter(item["license_recorded"] for item in manifest)),
        "license_note": "CC não identifica variante; ODBL é a string registrada na origem. Metadados não equivalem à verificação jurídica individual. Imagens não devem ser redistribuídas sem conferir os termos.",
        "negatives": 0,
        "unrecorded_license_included": args.include_unrecorded_license,
        "thumbnail_fallbacks": sum(
            item.get("download_url", item["source_url"]) != item["source_url"] for item in manifest
        ),
        "exif_normalized_images": sum(item.get("exif_orientation", 1) != 1 for item in manifest),
        "limitations": [
            "Sem negativos independentes revisados",
            "Sem capturas próprias de teste externo",
            "Agrupamento por dHash/origem; metadados não fornecem trajetos ou identidades de cena completas",
        ],
        "excluded_images": len(excluded),
        "download_failures": len(failures),
        "max_side": args.max_side,
    }
    artifacts = {
        "taco.json": manifest,
        "summary.json": summary,
        "excluded.json": excluded,
        "download_failures.json": failures,
        "class_mapping.json": [
            {"source_id": key, "source_name": value, "target_id": 0, "target_name": "residuo"}
            for key, value in categories.items()
        ],
    }
    for name, contents in artifacts.items():
        (args.manifests / name).write_text(
            json.dumps(contents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        print(
            "Algumas imagens não foram obtidas; consulte download_failures.json e execute novamente para retentar."
        )


if __name__ == "__main__":
    main()
