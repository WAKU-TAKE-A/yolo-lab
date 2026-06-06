import os
import sys
import json
import argparse
import shutil
import re
import urllib.parse

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

def write_yolo_yaml(dst_yaml_path, yaml_data, nc, names):
    yaml_data["nc"] = nc
    yaml_data["names"] = names
    try:
        import yaml
        with open(dst_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False, default_flow_style=None)
    except Exception:
        with open(dst_yaml_path, "w", encoding="utf-8") as f:
            for k, v in yaml_data.items():
                if k == "names":
                    if isinstance(names, list):
                        escaped = [n.replace("'", "\\'") for n in names]
                        f.write("names: [" + ", ".join([f"'{n}'" for n in escaped]) + "]\n")
                    else:
                        d_str = ", ".join([f"{cid}: '{n.replace(chr(39), chr(92)+chr(39))}'" for cid, n in names.items()])
                        f.write(f"names: {{{d_str}}}\n")
                elif k == "nc":
                    f.write(f"nc: {nc}\n")
                elif isinstance(v, (list, dict)):
                    f.write(f"{k}: {repr(v)}\n")
                else:
                    f.write(f"{k}: {v}\n")

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
    parser.add_argument("--remap", action="store_true")
    parser.add_argument("--images-root", type=str, default=None)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = {
        "conversion_pair": f"{args.fmt_from}->{args.fmt_to}",
        "dataset_path": args.dataset,
        "out_path": args.out,
        "image_count": 0,
        "annotation_count": 0,
        "task_count": 0,
        "annotations_converted": 0,
        "category_count": 0,
        "labels_written": 0,
        "images_copied": 0,
        "remap": args.remap,
        "class_map": {},
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

    fmt_from = args.fmt_from.lower()
    fmt_to = args.fmt_to.lower()

    if (fmt_from, fmt_to) not in [("yolo", "coco"), ("coco", "yolo"), ("labelstudio", "yolo")]:
        payload["errors"].append(f"Unsupported conversion pair: {fmt_from} -> {fmt_to}")
        die()

    ds_path = os.path.abspath(args.dataset)
    if not os.path.exists(ds_path):
        payload["errors"].append(f"Dataset path does not exist: {args.dataset}")
        die()

    out_path = os.path.abspath(args.out)
    if os.path.exists(out_path) and not args.force:
        payload["errors"].append(f"Output path already exists: {args.out}. Use --force to overwrite.")
        die()

    if fmt_from == "yolo" and fmt_to == "coco":
        # YOLO -> COCO implementation
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

    elif fmt_from == "coco" and fmt_to == "yolo":
        # COCO -> YOLO implementation
        if not os.path.isfile(ds_path):
            payload["errors"].append(f"Dataset must be a file path to COCO JSON: {args.dataset}")
            die()

        try:
            with open(ds_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
        except Exception as e:
            payload["errors"].append(f"Failed to read COCO JSON: {e}")
            die()
            
        if not isinstance(coco, dict) or "images" not in coco or "categories" not in coco or "annotations" not in coco:
            payload["errors"].append("Invalid COCO JSON format. Missing images, annotations, or categories.")
            die()

        coco_images = coco.get("images", [])
        coco_annotations = coco.get("annotations", [])
        coco_categories = coco.get("categories", [])
        
        payload["image_count"] = len(coco_images)
        payload["annotation_count"] = len(coco_annotations)
        payload["category_count"] = len(coco_categories)

        if os.path.exists(out_path) and args.force:
            try:
                if os.path.isdir(out_path):
                    shutil.rmtree(out_path)
                else:
                    os.remove(out_path)
            except Exception as e:
                payload["errors"].append(f"Failed to remove existing output path: {e}")
                die()

        os.makedirs(os.path.join(out_path, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "labels", "train"), exist_ok=True)

        if args.remap:
            sorted_cats = sorted(coco_categories, key=lambda c: c.get("id", 0))
            old_to_new_cat = {c.get("id"): idx for idx, c in enumerate(sorted_cats)}
            yolo_names = [c.get("name", f"class_{idx}") for idx, c in enumerate(sorted_cats)]
        else:
            old_to_new_cat = {c.get("id"): c.get("id") for c in coco_categories}
            yolo_names = {c.get("id"): c.get("name", f"class_{c.get('id')}") for c in coco_categories}

        ann_by_img = {}
        for ann in coco_annotations:
            img_id = ann.get("image_id")
            ann_by_img.setdefault(img_id, []).append(ann)

        images_copied = 0
        labels_written = 0

        coco_dir = os.path.dirname(ds_path)
        search_dirs = [coco_dir, os.path.dirname(coco_dir)]
        if args.images_root:
            search_dirs.insert(0, os.path.abspath(args.images_root))

        for img in coco_images:
            img_id = img.get("id")
            file_name = img.get("file_name", f"{img_id}.jpg")
            w = img.get("width")
            h = img.get("height")
            
            if w is None or h is None:
                payload["warnings"].append(f"Image {file_name} missing width or height. Skipping its annotations.")
                continue

            stem = os.path.splitext(os.path.basename(file_name))[0]
            
            src_img_path = None
            for sdir in search_dirs:
                cand1 = os.path.join(sdir, file_name)
                cand2 = os.path.join(sdir, os.path.basename(file_name))
                if os.path.isfile(cand1):
                    src_img_path = cand1
                    break
                elif os.path.isfile(cand2):
                    src_img_path = cand2
                    break
                    
            dst_img_path = os.path.join(out_path, "images", "train", os.path.basename(file_name))
            if src_img_path:
                try:
                    shutil.copy2(src_img_path, dst_img_path)
                    images_copied += 1
                except Exception as e:
                    payload["warnings"].append(f"Failed to copy image {file_name}: {e}")
            else:
                payload["warnings"].append(f"Image file not found: {file_name}")

            anns = ann_by_img.get(img_id, [])
            if anns:
                dst_lbl_path = os.path.join(out_path, "labels", "train", stem + ".txt")
                try:
                    with open(dst_lbl_path, "w", encoding="utf-8") as f:
                        for ann in anns:
                            cat_id = ann.get("category_id")
                            bbox = ann.get("bbox") 
                            if cat_id in old_to_new_cat and bbox and len(bbox) == 4:
                                yolo_cid = old_to_new_cat[cat_id]
                                x_min, y_min, bw, bh = bbox
                                xc = (x_min + bw / 2.0) / float(w)
                                yc = (y_min + bh / 2.0) / float(h)
                                nw = bw / float(w)
                                nh = bh / float(h)
                                f.write(f"{yolo_cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                    labels_written += 1
                except Exception as e:
                    payload["warnings"].append(f"Failed to write label for {file_name}: {e}")

        yaml_data = {
            "train": "images/train",
            "val": "images/train"
        }
        nc = len(yolo_names) if isinstance(yolo_names, list) else len(coco_categories)
        try:
            write_yolo_yaml(os.path.join(out_path, "data.yaml"), yaml_data, nc, yolo_names)
        except Exception as e:
            payload["warnings"].append(f"Failed to write data.yaml: {e}")

        payload["images_copied"] = images_copied
        payload["labels_written"] = labels_written
        
    elif fmt_from == "labelstudio" and fmt_to == "yolo":
        # Label Studio -> YOLO implementation
        if not os.path.isfile(ds_path):
            payload["errors"].append(f"Dataset must be a file path to Label Studio JSON: {args.dataset}")
            die()

        try:
            with open(ds_path, "r", encoding="utf-8") as f:
                ls_data = json.load(f)
        except Exception as e:
            payload["errors"].append(f"Failed to read Label Studio JSON: {e}")
            die()
            
        if not isinstance(ls_data, list):
            payload["errors"].append("Invalid Label Studio JSON. Expected a list of tasks.")
            die()

        payload["task_count"] = len(ls_data)

        if os.path.exists(out_path) and args.force:
            try:
                if os.path.isdir(out_path):
                    shutil.rmtree(out_path)
                else:
                    os.remove(out_path)
            except Exception as e:
                payload["errors"].append(f"Failed to remove existing output path: {e}")
                die()
        elif os.path.exists(out_path):
            payload["errors"].append(f"Output path already exists: {args.out}. Use --force to overwrite.")
            die()

        os.makedirs(os.path.join(out_path, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "labels", "train"), exist_ok=True)

        user_classes = None
        if args.classes:
            user_classes = [c.strip() for c in args.classes.split(",") if c.strip()]

        unique_labels = set()
        for task in ls_data:
            anns = task.get("annotations", [])
            if not anns: continue
            results = anns[0].get("result", [])
            for res in results:
                if res.get("type") == "rectanglelabels":
                    val = res.get("value", {})
                    rlabels = val.get("rectanglelabels", [])
                    if rlabels:
                        unique_labels.add(rlabels[0])

        if user_classes:
            class_list = user_classes
        else:
            class_list = sorted(list(unique_labels))

        class_to_id = {c: i for i, c in enumerate(class_list)}
        payload["category_count"] = len(class_list)
        payload["class_map"] = {str(v): k for k, v in class_to_id.items()}

        ls_dir = os.path.dirname(ds_path)
        search_dirs = [ls_dir, os.path.dirname(ls_dir)]
        if args.images_root:
            search_dirs.insert(0, os.path.abspath(args.images_root))

        images_copied = 0
        labels_written = 0
        annotations_converted = 0

        for task in ls_data:
            task_id = task.get("id", "unknown_task")
            data = task.get("data", {})
            img_ref = data.get("image") or data.get("image_url")
            
            if not img_ref:
                payload["warnings"].append(f"Task {task_id} missing image reference. Skipping.")
                continue
                
            if img_ref.startswith("file://"):
                img_ref = img_ref[7:]
            if img_ref.startswith("/data/local-files/?d="):
                img_ref = img_ref[len("/data/local-files/?d="):]

            img_ref = urllib.parse.unquote(img_ref)
            file_name = os.path.basename(img_ref)
            stem = os.path.splitext(file_name)[0]
            if not stem:
                stem = str(task_id)
                file_name = f"{stem}.jpg"

            src_img_path = None
            for sdir in search_dirs:
                cand1 = os.path.join(sdir, img_ref)
                cand2 = os.path.join(sdir, file_name)
                if os.path.isfile(cand1): src_img_path = cand1; break
                elif os.path.isfile(cand2): src_img_path = cand2; break

            dst_img_path = os.path.join(out_path, "images", "train", file_name)
            if src_img_path:
                try:
                    shutil.copy2(src_img_path, dst_img_path)
                    images_copied += 1
                except Exception as e:
                    payload["warnings"].append(f"Failed to copy image {file_name}: {e}")
            else:
                payload["warnings"].append(f"Image file not found: {img_ref} for task {task_id}")

            anns = task.get("annotations", [])
            if not anns:
                payload["warnings"].append(f"Task {task_id} has no annotations.")
                continue

            if len(anns) > 1:
                payload["warnings"].append(f"Task {task_id} has multiple annotations. Using the first one.")

            results = anns[0].get("result", [])
            lines = []
            for res in results:
                rtype = res.get("type")
                if rtype != "rectanglelabels":
                    payload["warnings"].append(f"Task {task_id} has unsupported annotation type '{rtype}'. Skipping.")
                    continue
                val = res.get("value", {})
                rlabels = val.get("rectanglelabels", [])
                if not rlabels:
                    continue
                label = rlabels[0]
                if label not in class_to_id:
                    payload["warnings"].append(f"Task {task_id} unknown class '{label}'. Skipping.")
                    continue

                cid = class_to_id[label]
                x_pct = float(val.get("x", 0))
                y_pct = float(val.get("y", 0))
                w_pct = float(val.get("width", 0))
                h_pct = float(val.get("height", 0))

                xc = (x_pct + w_pct / 2.0) / 100.0
                yc = (y_pct + h_pct / 2.0) / 100.0
                nw = w_pct / 100.0
                nh = h_pct / 100.0
                
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                annotations_converted += 1
                
            if lines:
                dst_lbl_path = os.path.join(out_path, "labels", "train", stem + ".txt")
                try:
                    with open(dst_lbl_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                    labels_written += 1
                except Exception as e:
                    payload["warnings"].append(f"Failed to write label for task {task_id}: {e}")

        yaml_data = {
            "train": "images/train",
            "val": "images/train"
        }
        try:
            write_yolo_yaml(os.path.join(out_path, "data.yaml"), yaml_data, len(class_list), class_list)
        except Exception as e:
            payload["warnings"].append(f"Failed to write data.yaml: {e}")

        payload["images_copied"] = images_copied
        payload["labels_written"] = labels_written
        payload["annotations_converted"] = annotations_converted

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Done")

if __name__ == "__main__":
    main()
