import os
import sys
import json
import argparse
import shutil
import re

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

def check_yolo_layout(dataset_dir):
    try:
        has_yaml = False
        for f in os.listdir(dataset_dir):
            if f.lower() in ["data.yaml", "dataset.yaml"]:
                has_yaml = True
                break
        has_images_and_labels = os.path.exists(os.path.join(dataset_dir, "images")) and os.path.exists(os.path.join(dataset_dir, "labels"))
        has_splits = False
        for split in ["train", "val", "test"]:
            if os.path.exists(os.path.join(dataset_dir, split, "images")) and os.path.exists(os.path.join(dataset_dir, split, "labels")):
                has_splits = True
                break
        return has_yaml or has_images_and_labels or has_splits
    except Exception:
        return False

def parse_yaml_fallback(file_content):
    data = {}
    names_match = re.search(r'names:\s*(\[.*?\]|\{.*?\}|\n(?:\s+\d+:.*\n?)+)', file_content)
    if names_match:
        val = names_match.group(1).strip()
        if val.startswith('['):
            try:
                import ast
                data['names'] = ast.literal_eval(val)
            except Exception: pass
        elif val.startswith('{'):
            try:
                import ast
                data['names'] = ast.literal_eval(val)
            except Exception: pass
        else:
            names_dict = {}
            for line in val.split('\n'):
                line_m = re.match(r'\s*(\d+):\s*(.*)', line)
                if line_m:
                    data.setdefault('names', {})[int(line_m.group(1))] = line_m.group(2).strip().strip("'\"")
    return data

def find_all_images(dataset_dir):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        rel_dir = os.path.relpath(root, dataset_dir)
        parts = rel_dir.split(os.sep)
        is_images_path = "images" in [p.lower() for p in parts]
        has_images_folder = os.path.exists(os.path.join(dataset_dir, "images"))
        if has_images_folder and not is_images_path:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in image_exts:
                image_paths.append(os.path.join(root, f))
    return image_paths

def find_label_for_image(dataset_dir, rel_image_path):
    parts = rel_image_path.split(os.sep)
    images_indices = [i for i, p in enumerate(parts) if p.lower() == "images"]
    stem, _ = os.path.splitext(parts[-1])
    if images_indices:
        for idx in images_indices:
            lbl_parts = list(parts)
            lbl_parts[idx] = "labels" if parts[idx].islower() else "LABELS" if parts[idx].isupper() else "Labels"
            lbl_parts[-1] = stem + ".txt"
            rel_lbl_path = os.path.join(*lbl_parts)
            abs_lbl_path = os.path.join(dataset_dir, rel_lbl_path)
            if os.path.exists(abs_lbl_path):
                return rel_lbl_path, abs_lbl_path
    for root, dirs, files in os.walk(dataset_dir):
        rel_dir = os.path.relpath(root, dataset_dir)
        dir_parts = rel_dir.split(os.sep)
        if "labels" in [dp.lower() for dp in dir_parts]:
            candidate = stem + ".txt"
            if candidate in files:
                abs_path = os.path.join(root, candidate)
                return os.path.relpath(abs_path, dataset_dir), abs_path
    if images_indices:
        idx = images_indices[0]
        lbl_parts = list(parts)
        lbl_parts[idx] = "labels" if parts[idx].islower() else "Labels"
        lbl_parts[-1] = stem + ".txt"
        return os.path.join(*lbl_parts), os.path.join(dataset_dir, os.path.join(*lbl_parts))
    parent_dir = os.path.dirname(rel_image_path)
    parent_parts = parent_dir.split(os.sep)
    new_parts = []
    replaced = False
    for p in parent_parts:
        if p.lower() == "images":
            new_parts.append("labels")
            replaced = True
        else:
            new_parts.append(p)
    if not replaced:
        new_parts.append("labels")
    rel_lbl_path = os.path.join(*(new_parts + [stem + ".txt"]))
    return rel_lbl_path, os.path.join(dataset_dir, rel_lbl_path)

def get_image_size(img_path, fb_w, fb_h):
    if HAS_PIL:
        try:
            with Image.open(img_path) as img:
                return img.width, img.height
        except Exception:
            pass
    return fb_w, fb_h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="fmt_from", required=True)
    parser.add_argument("--to", dest="fmt_to", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = {
        "conversion_pair": f"{args.fmt_from}->{args.fmt_to}",
        "dataset_path": args.dataset,
        "out_path": args.out,
        "image_count": 0,
        "annotation_count": 0,
        "category_count": 0,
        "warnings": [],
        "errors": []
    }
    
    def die():
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("Errors:", file=sys.stderr)
            for e in payload["errors"]: print(f" - {e}", file=sys.stderr)
        sys.exit(1 if payload["errors"] else 0)

    if args.fmt_from.lower() != "yolo" or args.fmt_to.lower() != "coco":
        payload["errors"].append(f"Unsupported conversion pair: {args.fmt_from} -> {args.fmt_to}")
        die()

    ds_path = os.path.abspath(args.dataset)
    if not os.path.exists(ds_path):
        payload["errors"].append(f"Dataset directory does not exist: {args.dataset}")
        die()

    out_path = os.path.abspath(args.out)
    if os.path.exists(out_path) and not args.force:
        payload["errors"].append(f"Output file already exists: {args.out}. Use --force to overwrite.")
        die()

    if not check_yolo_layout(ds_path):
        payload["warnings"].append("Dataset path does not look like a standard YOLO dataset.")

    yaml_file = None
    for f in os.listdir(ds_path):
        if f.lower() in ["data.yaml", "dataset.yaml"]:
            yaml_file = f
            break
            
    class_map = {}
    if yaml_file:
        try:
            with open(os.path.join(ds_path, yaml_file), "r", encoding="utf-8") as f:
                content = f.read()
            try:
                import yaml
                yd = yaml.safe_load(content)
            except Exception:
                yd = parse_yaml_fallback(content)
            names = yd.get("names", [])
            if isinstance(names, list):
                class_map = {idx: name for idx, name in enumerate(names)}
            elif isinstance(names, dict):
                class_map = {int(k): v for k, v in names.items()}
        except Exception as e:
            payload["warnings"].append(f"Failed to read yaml: {e}")

    categories = []
    for k, v in sorted(class_map.items()):
        categories.append({"id": k, "name": v})

    images_data = []
    annotations_data = []
    image_paths = find_all_images(ds_path)
    
    img_id_counter = 1
    ann_id_counter = 1
    known_classes = set(class_map.keys())

    for img_path in sorted(image_paths):
        rel_img = os.path.relpath(img_path, ds_path)
        rel_img_fwd = rel_img.replace(os.sep, '/')
        
        w, h = get_image_size(img_path, args.image_width, args.image_height)
        if w is None or h is None:
            payload["errors"].append(f"Cannot determine dimensions for {rel_img}. Provide --image-width and --image-height.")
            die()
            
        images_data.append({
            "id": img_id_counter,
            "file_name": rel_img_fwd,
            "width": w,
            "height": h
        })
        
        rel_lbl, abs_lbl = find_label_for_image(ds_path, rel_img)
        if os.path.exists(abs_lbl):
            try:
                with open(abs_lbl, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            xc = float(parts[1])
                            yc = float(parts[2])
                            bw = float(parts[3])
                            bh = float(parts[4])
                            
                            if cid not in known_classes:
                                known_classes.add(cid)
                                categories.append({"id": cid, "name": f"class_{cid}"})
                                
                            abs_w = bw * w
                            abs_h = bh * h
                            x_min = (xc * w) - (abs_w / 2)
                            y_min = (yc * h) - (abs_h / 2)
                            
                            annotations_data.append({
                                "id": ann_id_counter,
                                "image_id": img_id_counter,
                                "category_id": cid,
                                "bbox": [round(x_min, 4), round(y_min, 4), round(abs_w, 4), round(abs_h, 4)],
                                "area": round(abs_w * abs_h, 4),
                                "iscrowd": 0,
                                "source_label": rel_lbl.replace(os.sep, '/')
                            })
                            ann_id_counter += 1
            except Exception as e:
                payload["warnings"].append(f"Error parsing label {rel_lbl}: {e}")
                
        img_id_counter += 1

    coco = {
        "info": {"description": "Converted from YOLO"},
        "images": images_data,
        "annotations": annotations_data,
        "categories": categories
    }
    
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)
    except Exception as e:
        payload["errors"].append(f"Failed to write output: {e}")
        die()
        
    payload["image_count"] = len(images_data)
    payload["annotation_count"] = len(annotations_data)
    payload["category_count"] = len(categories)
    
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Done")

if __name__ == "__main__":
    main()
