from pathlib import Path
import shutil

SOURCE = Path("final_dataset")
DEST = Path("training_dataset")

CLASSES = {
    0: "helmet",
    1: "mask",
    2: "person",
}


def polygon_to_bbox(values):
    """
    values = [class_id, x1, y1, x2, y2, ...]
    Returns YOLO detection format:
    [class_id, x_center, y_center, width, height]
    """

    class_id = int(values[0])
    coords = values[1:]

    xs = coords[0::2]
    ys = coords[1::2]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return [
        class_id,
        x_center,
        y_center,
        width,
        height,
    ]


def process_split(split):
    source_images = SOURCE / "images" / split
    source_labels = SOURCE / "labels" / split

    dest_images = DEST / "images" / split
    dest_labels = DEST / "labels" / split

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    polygon_count = 0
    detection_count = 0
    image_count = 0

    for image_path in source_images.iterdir():

        if not image_path.is_file():
            continue

        # Copy image
        shutil.copy2(
            image_path,
            dest_images / image_path.name
        )

        image_count += 1

        # Corresponding label
        label_path = source_labels / f"{image_path.stem}.txt"
        dest_label_path = dest_labels / label_path.name

        if not label_path.exists():
            print(f"WARNING: Missing label for {image_path.name}")
            continue

        output_lines = []

        for line in label_path.read_text().splitlines():

            if not line.strip():
                continue

            values = [float(x) for x in line.split()]

            # Already YOLO detection format
            if len(values) == 5:

                class_id = int(values[0])

                if class_id not in CLASSES:
                    raise ValueError(
                        f"Invalid class {class_id} in {label_path}"
                    )

                output = values

                detection_count += 1

            # Polygon segmentation format
            elif len(values) >= 7 and (len(values) - 1) % 2 == 0:

                class_id = int(values[0])

                if class_id not in CLASSES:
                    raise ValueError(
                        f"Invalid class {class_id} in {label_path}"
                    )

                output = polygon_to_bbox(values)

                polygon_count += 1

            else:
                raise ValueError(
                    f"Invalid annotation format in {label_path}: "
                    f"{len(values)} values"
                )

            # Validate normalized YOLO coordinates
            if not all(0 <= x <= 1 for x in output[1:]):
                raise ValueError(
                    f"Coordinates outside [0,1] in {label_path}"
                )

            output_lines.append(
                " ".join(
                    f"{x:.8f}" if i > 0 else str(int(x))
                    for i, x in enumerate(output)
                )
            )

        dest_label_path.write_text(
            "\n".join(output_lines) + "\n"
        )

    return image_count, detection_count, polygon_count


def write_yaml():
    yaml_content = """path: C:/Users/sarim/safety monitoring/training_dataset
train: images/train
val: images/val

names:
  0: helmet
  1: mask
  2: person
"""

    (DEST / "data.yaml").write_text(yaml_content)


def main():

    if DEST.exists():
        raise RuntimeError(
            "training_dataset already exists. "
            "Delete it manually only if you want to recreate it."
        )

    print("=" * 60)
    print("CREATING YOLO DETECTION DATASET")
    print("=" * 60)

    total_images = 0
    total_detection = 0
    total_polygons = 0

    for split in ["train", "val"]:

        images, detections, polygons = process_split(split)

        total_images += images
        total_detection += detections
        total_polygons += polygons

        print(f"\n{split.upper()}")
        print(f"Images copied       : {images}")
        print(f"Detection kept      : {detections}")
        print(f"Polygons converted  : {polygons}")

    write_yaml()

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(f"Total images        : {total_images}")
    print(f"Detection boxes     : {total_detection}")
    print(f"Polygons converted  : {total_polygons}")
    print(f"Total annotations   : {total_detection + total_polygons}")

    print("\nClasses:")
    for class_id, name in CLASSES.items():
        print(f"  {class_id}: {name}")

    print("\nOutput:")
    print(DEST.resolve())

    print("\nOriginal dataset was NOT modified.")


if __name__ == "__main__":
    main()