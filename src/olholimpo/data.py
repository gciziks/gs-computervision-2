from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path, PurePosixPath
from typing import Iterable

from PIL import Image, ImageOps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(file_name: str) -> Path:
    normalized = file_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or not path.parts
        or path.is_absolute()
        or any(part in {"..", "."} or ":" in part for part in path.parts)
    ):
        raise ValueError(f"Caminho de imagem inválido: {file_name!r}")
    return Path(*path.parts)


def coco_bbox_to_yolo(
    bbox: Iterable[float], width: int, height: int
) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in bbox)
    if (
        len(values) != 4
        or width <= 0
        or height <= 0
        or not all(math.isfinite(value) for value in values)
    ):
        raise ValueError("Dimensões ou caixa inválidas")
    x, y, box_width, box_height = values
    if (
        box_width <= 0
        or box_height <= 0
        or x < 0
        or y < 0
        or x + box_width > width + 1e-4
        or y + box_height > height + 1e-4
    ):
        raise ValueError("Caixa fora dos limites da imagem")
    return (
        (x + box_width / 2) / width,
        (y + box_height / 2) / height,
        box_width / width,
        box_height / height,
    )


def difference_hash(image: Image.Image) -> int:
    resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = [resized.getpixel((column, row)) for row in range(8) for column in range(9)]
    result = 0
    for row in range(8):
        for column in range(8):
            result = (result << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return result


def normalize_annotation_image(
    image: Image.Image, width: int, height: int, allow_resized: bool = False
) -> Image.Image:
    normalized = ImageOps.exif_transpose(image)
    ratio_difference = abs((normalized.width / normalized.height) / (width / height) - 1)
    if normalized.size != (width, height) and (not allow_resized or ratio_difference > 0.01):
        raise ValueError(f"Dimensões diferentes da anotação: {normalized.size}")
    return normalized


def group_similar(records: list[dict], distance: int = 6) -> list[list[dict]]:
    if not 0 <= distance <= 64:
        raise ValueError("Distância de hash deve estar entre 0 e 64")
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, record in enumerate(records):
        for other_index in range(index):
            other = records[other_index]
            same_source = record.get("source_url") and record["source_url"] == other.get(
                "source_url"
            )
            same_bytes = record["sha256"] == other["sha256"]
            similar = (int(record["dhash"], 16) ^ int(other["dhash"], 16)).bit_count() <= distance
            if same_source or same_bytes or similar:
                parents[find(index)] = find(other_index)
    grouped: dict[int, list[dict]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    return list(grouped.values())


def assign_splits(groups: list[list[dict]], seed: int = 42) -> list[dict]:
    if len(groups) < 3:
        raise ValueError("São necessários pelo menos três grupos independentes")
    ordered = sorted(groups, key=lambda group: min(record["image_id"] for record in group))
    random.Random(seed).shuffle(ordered)
    total = sum(len(group) for group in ordered)
    targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
    counts = dict.fromkeys(targets, 0)
    result = []
    for group in ordered:
        split = max(targets, key=lambda name: (targets[name] - counts[name]) / targets[name])
        group_id = "g" + str(min(record["image_id"] for record in group))
        for record in group:
            result.append({**record, "group_id": group_id, "split": split})
        counts[split] += len(group)
    return sorted(result, key=lambda record: record["image_id"])
