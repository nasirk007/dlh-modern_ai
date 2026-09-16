#!/usr/bin/env python3
"""Turn Pascal VOC 2012 labels into the YOLO format.
Keeps only person, car and bicycle. Copies the images and writes one
.txt label file for each image."""
import os
import shutil
import xml.etree.ElementTree as ET


VOC_ROOT = "VOC2012"
OUTPUT_ROOT = "datasets/detection"

# The order gives the class id: person 0, car 1, bicycle 2.
# It must be the same as the names list in data.yaml.
CLASSES = ["person", "car", "bicycle"]


def parse_annotation(image_id):
    """Read one VOC XML file and return its YOLO label lines.
    Keeps only our classes. Turns pixel corners into center and size,
    divided by the image size. Empty list if nothing matches."""
    xml_path = os.path.join(VOC_ROOT, "Annotations", image_id + ".xml")
    root = ET.parse(xml_path).getroot()

    # The image size is inside the XML, so we never open the jpg.
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASSES:
            continue

        # Use float, not int: some VOC boxes have decimal values.
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
            CLASSES.index(name),
            ((xmin + xmax) / 2) / img_w,
            ((ymin + ymax) / 2) / img_h,
            (xmax - xmin) / img_w,
            (ymax - ymin) / img_h,
        ))
    return lines


def prepare_split(split):
    """Copy the images and write the labels for one split.
    Creates the sample list from the VOC split list if it is missing."""
    samples = split + "_samples.txt"
    if not os.path.exists(samples):
        voc_list = os.path.join(
            VOC_ROOT, "ImageSets", "Main", split + ".txt")
        with open(voc_list, "r") as f:
            all_ids = [line.strip() for line in f if line.strip()]
        with open(samples, "w") as f:
            f.write("\n".join(i for i in all_ids if parse_annotation(i)))
            f.write("\n")

    with open(samples, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    image_dir = os.path.join(OUTPUT_ROOT, "images", split)
    label_dir = os.path.join(OUTPUT_ROOT, "labels", split)
    kept = 0

    for image_id in ids:
        lines = parse_annotation(image_id)
        if not lines:
            continue

        # The image and its label must have the same name. This is how
        # YOLO finds the label of an image.
        shutil.copyfile(
            os.path.join(VOC_ROOT, "JPEGImages", image_id + ".jpg"),
            os.path.join(image_dir, image_id + ".jpg"))
        with open(os.path.join(label_dir, image_id + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        kept += 1

    print("{}: {} images, {} skipped".format(
        split, kept, len(ids) - kept))


if __name__ == "__main__":
    prepare_split("train")
    prepare_split("val")
