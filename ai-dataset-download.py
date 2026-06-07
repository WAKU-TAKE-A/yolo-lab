import argparse
import concurrent.futures
import json
import os
import random
import shutil
import sys
import tempfile
import urllib.request
import zipfile


SMALL_PRESETS = {
    "coco8": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
        "root": "coco8",
    },
    "coco8-seg": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-seg.zip",
        "root": "coco8-seg",
    },
    "coco128": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "root": "coco128",
    },
    "coco128-seg": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip",
        "root": "coco128-seg",
    },
}

SAMPLE_IMAGES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}

FULL_COCO_URLS = {
    "labels": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def emit(payload, as_json):
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        for key, value in payload.items():
            if key in ("warnings", "errors"):
                continue
            print(f"{key}: {value}")
        if payload.get("warnings"):
            print("Warnings:")
            for warning in payload["warnings"]:
                print(f"  - {warning}")
        if payload.get("errors"):
            print("Errors:", file=sys.stderr)
            for error in payload["errors"]:
                print(f"  - {error}", file=sys.stderr)


def download_file(url, dest_path, timeout=60):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return "exists"
    urllib.request.urlretrieve(url, dest_path)
    return "downloaded"


def safe_replace_dir(src_dir, out_dir, force):
    if os.path.exists(out_dir):
        if not force:
            raise FileExistsError(f"Output path already exists: {out_dir}. Use --force to overwrite.")
        shutil.rmtree(out_dir)
    os.makedirs(os.path.dirname(out_dir) or ".", exist_ok=True)
    shutil.move(src_dir, out_dir)


def download_zip_preset(preset, out_dir, force):
    spec = SMALL_PRESETS[preset]
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, f"{preset}.zip")
        download_file(spec["url"], zip_path)
        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        expected_root = os.path.join(extract_dir, spec["root"])
        if os.path.isdir(expected_root):
            safe_replace_dir(expected_root, out_dir, force)
        else:
            safe_replace_dir(extract_dir, out_dir, force)


def download_and_extract_zip(url, dest_dir, zip_name):
    os.makedirs(dest_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, zip_name)
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)


def download_full_coco(out_dir, force, include_test, payload):
    if os.path.exists(out_dir):
        if not force:
            raise FileExistsError(f"Output path already exists: {out_dir}. Use --force to overwrite.")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    parent_dir = os.path.dirname(out_dir) or "."
    download_and_extract_zip(FULL_COCO_URLS["labels"], parent_dir, "coco2017labels-segments.zip")

    images_dir = os.path.join(out_dir, "images")
    download_and_extract_zip(FULL_COCO_URLS["train2017"], images_dir, "train2017.zip")
    download_and_extract_zip(FULL_COCO_URLS["val2017"], images_dir, "val2017.zip")
    if include_test:
        download_and_extract_zip(FULL_COCO_URLS["test2017"], images_dir, "test2017.zip")

    payload["dataset_path"] = os.path.abspath(out_dir)
    payload["source_urls"] = {
        "labels": FULL_COCO_URLS["labels"],
        "train2017": FULL_COCO_URLS["train2017"],
        "val2017": FULL_COCO_URLS["val2017"],
        "test2017": FULL_COCO_URLS["test2017"] if include_test else None,
    }


def download_sample_images(out_dir):
    image_dir = os.path.join(out_dir, "images")
    results = []
    for name, url in SAMPLE_IMAGES.items():
        dest = os.path.join(image_dir, name)
        try:
            status = download_file(url, dest)
            results.append({"name": name, "status": status, "path": dest})
        except Exception as exc:
            results.append({"name": name, "status": "failed", "error": str(exc), "path": dest})
    return results


def resolve_coco_classes(classes_arg):
    if not classes_arg:
        raise ValueError("--classes is required for coco-train-subset")
    name_to_id = {name.lower(): idx for idx, name in enumerate(COCO_NAMES)}
    resolved = []
    for raw in [p.strip() for p in classes_arg.split(",") if p.strip()]:
        try:
            cid = int(raw)
        except ValueError:
            key = raw.lower()
            if key not in name_to_id:
                raise ValueError(f"Unknown COCO class name: {raw}")
            cid = name_to_id[key]
        if cid < 0 or cid >= len(COCO_NAMES):
            raise ValueError(f"Invalid COCO class id: {cid}")
        if cid not in resolved:
            resolved.append(cid)
    return resolved


def read_filtered_label(label_path, target_classes, class_mapping):
    class_counts = {}
    new_lines = []
    with open(label_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if cid not in target_classes:
                continue
            class_counts[cid] = class_counts.get(cid, 0) + 1
            parts[0] = str(class_mapping[cid])
            new_lines.append(" ".join(parts) + "\n")
    return new_lines, class_counts


def select_coco_subset(args, payload):
    labels_dir = os.path.abspath(args.labels)
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory does not exist: {args.labels}")

    target_classes = resolve_coco_classes(args.classes)
    class_mapping = {cid: cid if args.no_remap else idx for idx, cid in enumerate(target_classes)}
    payload["classes_resolved"] = target_classes
    payload["class_mapping"] = {str(k): v for k, v in class_mapping.items()}

    candidates = []
    matched_per_class = {str(cid): 0 for cid in target_classes}
    matched_ann_per_class = {str(cid): 0 for cid in target_classes}

    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
    label_files.sort()
    payload["labels_scanned"] = len(label_files)

    for file_name in label_files:
        label_path = os.path.join(labels_dir, file_name)
        new_lines, class_counts = read_filtered_label(label_path, target_classes, class_mapping)
        if not class_counts:
            continue
        for cid, count in class_counts.items():
            matched_per_class[str(cid)] += 1
            matched_ann_per_class[str(cid)] += count
        candidates.append({
            "label_name": file_name,
            "image_name": os.path.splitext(file_name)[0] + ".jpg",
            "label_path": label_path,
            "new_lines": new_lines,
            "class_counts": class_counts,
        })

    if args.sample_strategy == "random":
        rng = random.Random(args.seed)
        rng.shuffle(candidates)

    selected = []
    selected_per_class = {str(cid): 0 for cid in target_classes}
    max_total = args.max_images
    max_per_class = args.max_images_per_class

    for cand in candidates:
        if max_total is not None and len(selected) >= max_total:
            break
        if max_per_class is not None:
            would_exceed = False
            for cid in cand["class_counts"]:
                if selected_per_class[str(cid)] >= max_per_class:
                    would_exceed = True
                    break
            if would_exceed:
                continue
        selected.append(cand)
        for cid in cand["class_counts"]:
            selected_per_class[str(cid)] += 1

    payload["matched_images_total"] = len(candidates)
    payload["matched_annotations_total"] = sum(matched_ann_per_class.values())
    payload["per_class_image_count"] = matched_per_class
    payload["per_class_annotation_count"] = matched_ann_per_class
    payload["selected_images_total"] = len(selected)
    payload["selected_per_class_image_count"] = selected_per_class
    payload["selected_images"] = [cand["image_name"] for cand in selected]

    if args.stats_only:
        return selected

    out_dir = os.path.abspath(args.out)
    if os.path.exists(out_dir):
        if not args.force:
            raise FileExistsError(f"Output path already exists: {args.out}. Use --force to overwrite.")
        shutil.rmtree(out_dir)

    image_dir = os.path.join(out_dir, "images", "train2017")
    label_dir = os.path.join(out_dir, "labels", "train2017")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    def fetch(cand):
        image_path = os.path.join(image_dir, cand["image_name"])
        url = f"http://images.cocodataset.org/train2017/{cand['image_name']}"
        status = download_file(url, image_path)
        label_out = os.path.join(label_dir, cand["label_name"])
        with open(label_out, "w", encoding="utf-8") as f:
            f.writelines(cand["new_lines"])
        return status

    downloaded = 0
    existing = 0
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(fetch, cand): cand for cand in selected}
        for future in concurrent.futures.as_completed(future_map):
            cand = future_map[future]
            try:
                status = future.result()
                if status == "downloaded":
                    downloaded += 1
                else:
                    existing += 1
            except Exception as exc:
                failed.append({"image": cand["image_name"], "error": str(exc)})

    yaml_path = os.path.join(out_dir, "data.yaml")
    names = [COCO_NAMES[cid] for cid in target_classes] if not args.no_remap else COCO_NAMES
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: images/train2017\n")
        f.write("val: images/train2017\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names: [" + ", ".join(repr(name) for name in names) + "]\n")

    payload["images_downloaded"] = downloaded
    payload["images_existing"] = existing
    payload["download_failures"] = failed
    payload["data_yaml"] = yaml_path
    return selected


def main():
    parser = argparse.ArgumentParser(description="Download sample images and YOLO dataset presets for YOLO-Lab")
    parser.add_argument("--preset", required=True, choices=[
        "sample-images", "coco8", "coco8-seg", "coco128", "coco128-seg", "coco", "coco-train-subset"
    ])
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory for dataset downloads")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    parser.add_argument("--labels", help="COCO YOLO labels/train2017 directory for coco-train-subset")
    parser.add_argument("--classes", help="Comma-separated COCO class IDs or names for coco-train-subset")
    parser.add_argument("--max-images", type=int, help="Maximum total images for coco-train-subset")
    parser.add_argument("--max-images-per-class", type=int, help="Maximum images per class for coco-train-subset")
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--no-remap", action="store_true")
    parser.add_argument("--allow-large", action="store_true", help="Required for full COCO download guidance")
    parser.add_argument("--include-test", action="store_true", help="Also download COCO test2017 for --preset coco")

    args = parser.parse_args()

    payload = {
        "preset": args.preset,
        "out": args.out,
        "warnings": [],
        "errors": [],
    }

    try:
        if args.preset == "sample-images":
            payload["files"] = download_sample_images(args.out)
        elif args.preset in SMALL_PRESETS:
            download_zip_preset(args.preset, os.path.abspath(args.out), args.force)
            payload["dataset_path"] = os.path.abspath(args.out)
            payload["source_url"] = SMALL_PRESETS[args.preset]["url"]
        elif args.preset == "coco":
            if not args.allow_large:
                raise ValueError("Full COCO is very large. Re-run with --allow-large after confirming disk/time budget.")
            download_full_coco(os.path.abspath(args.out), args.force, args.include_test, payload)
        elif args.preset == "coco-train-subset":
            select_coco_subset(args, payload)
        else:
            raise ValueError(f"Unsupported preset: {args.preset}")
    except Exception as exc:
        payload["errors"].append(str(exc))
        emit(payload, args.json)
        sys.exit(0 if args.json else 1)

    emit(payload, args.json)


if __name__ == "__main__":
    main()
