#!/usr/bin/env python3
"""Prepare a filtered Pascal VOC dataset in YOLOv8 format."""
import argparse
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


VOC_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
    "VOCtrainval_11-May-2012.tar"
)
CLASSES = {"person": 0, "car": 1, "bicycle": 2}
DATASET_ROOT = Path("datasets/detection")


def read_sample_names(path):
    """Read image identifiers from a sample-list file."""
    with path.open(encoding="utf-8") as sample_file:
        return [
            Path(line.strip().split()[0]).stem
            for line in sample_file
            if line.strip()
        ]


def find_sample_file(filename):
    """Find a sample-list file beside the script/in the current directory."""
    candidates = (Path(filename), Path(__file__).resolve().parent / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing {filename}; place it beside 0-prep_data.py."
    )


def extract_archive(archive_path, destination):
    """Extract a Pascal VOC archive and return its VOC2012 directory."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(destination)
    return find_voc_root(destination)


def find_voc_root(path):
    """Find the VOC2012 directory below a supplied path."""
    candidates = [path / "VOCdevkit" / "VOC2012"]
    candidates.extend(path.glob("*/VOCdevkit/VOC2012"))
    for candidate in candidates:
        if (candidate / "JPEGImages").is_dir():
            return candidate
    raise FileNotFoundError("Cld not find VOCdevkit/VOC2012 in supplied path")


def get_voc_root(voc_root, archive):
    """Resolve an extracted VOC directory or download and extract archive."""
    if voc_root:
        return find_voc_root(Path(voc_root))
    if archive:
        return extract_archive(Path(archive), Path(".voc_data"))

    local_archive = Path("VOCtrainval_11-May-2012.tar")
    if local_archive.is_file():
        return extract_archive(local_archive, Path(".voc_data"))

    download_path = Path(".voc_data") / "VOCtrainval_11-May-2012.tar"
    download_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Pascal VOC 2012...")
    urllib.request.urlretrieve(VOC_URL, download_path)
    return extract_archive(download_path, download_path.parent)


def parse_annotation(annotation_path):
    """Return image dimensions and selected Pascal VOC objects."""
    root = ET.parse(annotation_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    objects = []
    for item in root.findall("object"):
        class_name = item.findtext("name")
        if class_name not in CLASSES:
            continue
        box = item.find("bndbox")
        coordinates = [
            float(box.findtext("xmin")),
            float(box.findtext("ymin")),
            float(box.findtext("xmax")),
            float(box.findtext("ymax")),
        ]
        objects.append((CLASSES[class_name], coordinates))
    return width, height, objects


def convert_box(box, width, height):
    """Convert a Pascal VOC box to normalized YOLO coordinates."""
    xmin, ymin, xmax, ymax = box
    return (
        ((xmin + xmax) / 2) / width,
        ((ymin + ymax) / 2) / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def prepare_split(voc_root, names, split):
    """Copy selected images and write their filtered YOLO label files."""
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_name in names:
        image_path = voc_root / "JPEGImages" / f"{image_name}.jpg"
        annotation_path = voc_root / "Annotations" / f"{image_name}.xml"
        if not image_path.is_file() or not annotation_path.is_file():
            raise FileNotFoundError(f"Missing VOC files for {image_name}.")
        shutil.copy2(image_path, image_dir / image_path.name)
        width, height, objects = parse_annotation(annotation_path)
        label_path = label_dir / f"{image_name}.txt"
        with label_path.open("w", encoding="utf-8") as label_file:
            for class_id, box in objects:
                values = convert_box(box, width, height)
                label_file.write(
                    f"{class_id} "
                    + " ".join(f"{value:.6f}" for value in values)
                    + "\n"
                )


def write_data_yaml():
    """Write the required YOLOv8 dataset configuration."""
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "data.yaml").write_text(
        "path: datasets/detection/\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 3\n"
        'names: ["person", "car", "bicycle"]\n',
        encoding="utf-8",
    )


def main():
    """Prepare the filtered YOLOv8 dataset from Pascal VOC 2012."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    train_file = find_sample_file("train_samples.txt")
    val_file = find_sample_file("val_samples.txt")
    voc_root = get_voc_root(args.voc_root, args.archive)
    prepare_split(voc_root, read_sample_names(train_file), "train")
    prepare_split(voc_root, read_sample_names(val_file), "val")
    write_data_yaml()
    print(f"Dataset prepared in {DATASET_ROOT}")


if __name__ == "__main__":
    main()
