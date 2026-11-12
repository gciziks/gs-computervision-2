import math
from pathlib import Path

import pytest
from PIL import Image

from olholimpo.data import (
    assign_splits,
    coco_bbox_to_yolo,
    difference_hash,
    group_similar,
    normalize_annotation_image,
    safe_relative_path,
)


def test_bbox_conversion_preserves_full_image_and_off_center_object():
    assert coco_bbox_to_yolo([0, 0, 200, 100], 200, 100) == (0.5, 0.5, 1.0, 1.0)
    assert coco_bbox_to_yolo([20, 10, 40, 30], 200, 100) == (0.2, 0.25, 0.2, 0.3)


@pytest.mark.parametrize(
    "bbox", [[-1, 0, 20, 20], [0, 0, 0, 10], [190, 0, 20, 10], [0, 0, math.nan, 10], [0, 0, 1]]
)
def test_invalid_boxes_are_rejected_instead_of_becoming_training_labels(bbox):
    with pytest.raises(ValueError):
        coco_bbox_to_yolo(bbox, 200, 100)


@pytest.mark.parametrize(
    "name",
    [
        "../secret.jpg",
        "C:/secret.jpg",
        "/secret.jpg",
        "batch/../../secret.jpg",
        "..\\secret.jpg",
        "",
        ".",
    ],
)
def test_dataset_paths_cannot_escape_destination(name):
    with pytest.raises(ValueError):
        safe_relative_path(name)


def test_safe_dataset_path():
    assert safe_relative_path("batch_1/000006.jpg") == Path("batch_1/000006.jpg")


def test_transitive_similar_images_remain_in_same_split():
    records = [
        {"image_id": 1, "sha256": "a", "dhash": "0000000000000000"},
        {"image_id": 2, "sha256": "b", "dhash": "0000000000000001"},
        {"image_id": 3, "sha256": "c", "dhash": "0000000000000003"},
        {"image_id": 4, "sha256": "d", "dhash": "ffffffffffffffff"},
        {"image_id": 5, "sha256": "e", "dhash": "aaaaaaaaaaaaaaaa"},
        {"image_id": 6, "sha256": "f", "dhash": "5555555555555555"},
    ]
    groups = group_similar(records, distance=1)
    manifest = assign_splits(groups, seed=42)
    assert len({record["split"] for record in manifest if record["image_id"] in {1, 2, 3}}) == 1
    assert len({record["split"] for record in manifest}) == 3
    assert manifest == assign_splits(list(reversed(groups)), seed=42)


def test_identical_source_is_grouped_even_when_hash_is_different():
    records = [
        {
            "image_id": index,
            "sha256": str(index),
            "dhash": value,
            "source_url": "https://example.test/image",
        }
        for index, value in enumerate(["0", "ffffffffffffffff"])
    ]
    assert len(group_similar(records, distance=0)) == 1


def test_dhash_is_stable_for_same_pixels():
    image = Image.new("RGB", (100, 100), "white")
    assert difference_hash(image) == difference_hash(image.copy())


def test_exif_rotation_matches_annotation_dimensions_before_hash_and_boxes():
    image = Image.new("RGB", (40, 20), "white")
    image.putpixel((0, 0), (255, 0, 0))
    image.getexif()[274] = 6
    normalized = normalize_annotation_image(image, width=20, height=40)
    assert normalized.size == (20, 40)
    assert normalized.getpixel((19, 0)) == (255, 0, 0)
    assert coco_bbox_to_yolo([0, 0, 20, 40], *normalized.size) == (0.5, 0.5, 1, 1)


def test_thumbnail_resize_is_accepted_only_with_same_aspect_ratio():
    assert normalize_annotation_image(
        Image.new("RGB", (640, 320)), 4000, 2000, allow_resized=True
    ).size == (640, 320)
    with pytest.raises(ValueError):
        normalize_annotation_image(Image.new("RGB", (640, 400)), 4000, 2000, allow_resized=True)
