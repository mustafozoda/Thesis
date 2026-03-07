import json
import csv
from pathlib import Path
import cv2


from pathlib import Path

DATA_ROOT = Path(r"C:/Users/User/Desktop/LaboroTomato/dataset/original")

IMAGES_TRAIN = DATA_ROOT / "images" / "train"
IMAGES_TEST = DATA_ROOT / "images" / "test"

ANNS_TRAIN = DATA_ROOT / "ann" / "train"
ANNS_TEST = DATA_ROOT / "ann" / "test"

OUT_ROOT = Path(r"./dataset/cropped")

MARGIN_RATIO = 0.25     # 25% margin around bbox
MIN_CROP_SIZE = 64      # discard tiny crops


# Label mapping: b_/l_ -> 3 classes

def map_label(class_title: str) -> str:
    ct = class_title.lower()
    if "green" in ct and "half" not in ct and "fully" not in ct:
        return "green"
    if "half_ripened" in ct or "half" in ct:
        return "half_ripened"
    if "fully_ripened" in ct or "fully" in ct:
        return "fully_ripened"
    return "unknown"


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def bbox_from_polygon(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def expand_bbox(x1, y1, x2, y2, w, h, margin_ratio):
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    nx1 = clamp(x1 - mx, 0, w - 1)
    ny1 = clamp(y1 - my, 0, h - 1)
    nx2 = clamp(x2 + mx, 0, w - 1)
    ny2 = clamp(y2 + my, 0, h - 1)
    return nx1, ny1, nx2, ny2


def process_split(split_name: str, images_dir: Path, anns_dir: Path, writer):
    ann_files = sorted(list(anns_dir.rglob("*.json")))
    print(f"[{split_name}] anns_dir = {anns_dir} | found {len(ann_files)} json files")

    if not ann_files:
        raise RuntimeError(f"No annotation files found in: {anns_dir}")

    saved = 0
    skipped_unknown = 0
    skipped_small = 0
    missing_images = 0

    for ann_path in ann_files:
        # IMG_0984.jpg.json -> IMG_0984.jpg
        image_name = ann_path.name.replace(".json", "")
        image_path = images_dir / image_name

        if not image_path.exists():
            missing_images += 1
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            missing_images += 1
            continue

        h, w = img.shape[:2]

        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        objects = ann.get("objects", [])

        for obj in objects:
            label = map_label(obj.get("classTitle", ""))
            if label == "unknown":
                skipped_unknown += 1
                continue

            pts = obj.get("points", {}).get("exterior", [])
            if not pts or len(pts) < 3:
                continue

            x1, y1, x2, y2 = bbox_from_polygon(pts)
            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, MARGIN_RATIO)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
                skipped_small += 1
                continue

            crop = img[y1:y2, x1:x2]

            out_dir = OUT_ROOT / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)

            obj_id = obj.get("id", "na")
            out_name = f"{image_path.stem}__obj{obj_id}__{label}.jpg"
            out_path = out_dir / out_name

            cv2.imwrite(str(out_path), crop)

            writer.writerow([
                str(out_path), label, split_name,
                image_name, obj_id,
                x1, y1, x2, y2,
                w, h
            ])
            saved += 1

    return saved, skipped_unknown, skipped_small, missing_images


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_ROOT / "crops.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "crop_path", "label", "split",
            "orig_image", "object_id",
            "x1", "y1", "x2", "y2",
            "orig_w", "orig_h"
        ])

        train_stats = process_split("train", IMAGES_TRAIN, ANNS_TRAIN, writer)
        test_stats = process_split("test", IMAGES_TEST, ANNS_TEST, writer)

    print("Done.")
    print(
        f"Train: saved={train_stats[0]}, skipped_unknown={train_stats[1]}, skipped_small={train_stats[2]}, missing_images={train_stats[3]}")
    print(
        f"Test : saved={test_stats[0]},  skipped_unknown={test_stats[1]},  skipped_small={test_stats[2]},  missing_images={test_stats[3]}")
    print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
