import os
import sys
import json
import argparse
import shutil
import re

def check_yolo_layout(dataset_dir):
    try:
        # Check for data.yaml or dataset.yaml
        has_yaml = False
        for f in os.listdir(dataset_dir):
            if f.lower() in ["data.yaml", "dataset.yaml"]:
                has_yaml = True
                break
                
        # Check common folder patterns:
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
    
    # Simple regex parsing for names mapping/list
    names_match = re.search(r'names:\s*(\[.*?\]|\{.*?\}|\n(?:\s+\d+:.*\n?)+)', file_content)
    if names_match:
        val = names_match.group(1).strip()
        if val.startswith('['):
            try:
                import ast
                data['names'] = ast.literal_eval(val)
            except Exception:
                pass
        elif val.startswith('{'):
            try:
                import ast
                data['names'] = ast.literal_eval(val)
            except Exception:
                pass
        else:
            names_dict = {}
            for line in val.split('\n'):
                line_m = re.match(r'\s*(\d+):\s*(.*)', line)
                if line_m:
                    k = int(line_m.group(1))
                    v = line_m.group(2).strip().strip("'\"")
                    names_dict[k] = v
            if names_dict:
                data['names'] = names_dict

    for key in ['train', 'val', 'test', 'nc']:
        m = re.search(rf'^{key}:\s*(.*)', file_content, re.MULTILINE)
        if m:
            val = m.group(1).strip().strip("'\"")
            if key == 'nc':
                try:
                    data[key] = int(val)
                except ValueError:
                    pass
            else:
                data[key] = val
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
                rel_path = os.path.relpath(abs_path, dataset_dir)
                return rel_path, abs_path
                
    if images_indices:
        idx = images_indices[0]
        lbl_parts = list(parts)
        lbl_parts[idx] = "labels" if parts[idx].islower() else "Labels"
        lbl_parts[-1] = stem + ".txt"
        rel_lbl_path = os.path.join(*lbl_parts)
        abs_lbl_path = os.path.join(dataset_dir, rel_lbl_path)
        return rel_lbl_path, abs_lbl_path
        
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
    abs_lbl_path = os.path.join(dataset_dir, rel_lbl_path)
    return rel_lbl_path, abs_lbl_path

def resolve_classes(classes_arg, class_map):
    resolved_ids = []
    errors = []
    
    ci_map = {v.lower(): k for k, v in class_map.items()}
    
    parts = [p.strip() for p in classes_arg.split(",") if p.strip()]
    for part in parts:
        try:
            val_int = int(part)
            if val_int in class_map or val_int >= 0:
                resolved_ids.append(val_int)
            else:
                errors.append(f"Invalid class ID: {part} (must be non-negative)")
        except ValueError:
            name_lower = part.lower()
            if name_lower in ci_map:
                resolved_ids.append(ci_map[name_lower])
            else:
                errors.append(f"Class name '{part}' not found in dataset classes.")
                
    seen = set()
    unique_ids = []
    for r in resolved_ids:
        if r not in seen:
            seen.add(r)
            unique_ids.append(r)
            
    return unique_ids, errors

def filter_and_edit_annotations(label_path, target_classes, class_mapping):
    lines_scanned = 0
    lines_extracted = 0
    new_lines = []
    
    if not os.path.exists(label_path):
        return new_lines, lines_scanned, lines_extracted
        
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            stripped = line.rstrip('\r\n')
            if not stripped:
                new_lines.append(line)
                continue
            if stripped.startswith('#') or stripped.startswith('//'):
                new_lines.append(line)
                continue
                
            tokens = stripped.split()
            if not tokens:
                new_lines.append(line)
                continue
                
            first_token = tokens[0]
            try:
                class_id = int(first_token)
                lines_scanned += 1
            except ValueError:
                new_lines.append(line)
                continue
                
            if class_id in target_classes:
                new_class_id = class_mapping[class_id]
                tokens[0] = str(new_class_id)
                suffix = line[len(stripped):]
                new_line = " ".join(tokens) + suffix
                new_lines.append(new_line)
                lines_extracted += 1
    except Exception:
        pass
        
    return new_lines, lines_scanned, lines_extracted

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
                        escaped_names = [name.replace("'", "\\'") for name in names]
                        names_str = ", ".join([f"'{n}'" for n in escaped_names])
                        f.write(f"names: [{names_str}]\n")
                    else:
                        # dict representation
                        dict_str = ", ".join(["{}: '{}'".format(cid, name.replace("'", "\\'")) for cid, name in names.items()])
                        f.write(f"names: {{{dict_str}}}\n")
                elif k == "nc":
                    f.write(f"nc: {nc}\n")
                elif isinstance(v, (list, dict)):
                    f.write(f"{k}: {repr(v)}\n")
                else:
                    f.write(f"{k}: {v}\n")

def main():
    parser = argparse.ArgumentParser(description="AI-first YOLO category extraction utility")
    parser.add_argument("--dataset", type=str, required=True, help="Path to source YOLO dataset")
    parser.add_argument("--out", type=str, required=True, help="Path to output dataset directory")
    parser.add_argument("--classes", type=str, required=True, help="Comma-separated class IDs or names to extract")
    parser.add_argument("--keep-empty-images", action="store_true", help="Keep all images even if they have no target annotations")
    parser.add_argument("--no-remap", action="store_true", help="Preserve original class IDs instead of remapping to contiguous indices")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing output directory")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    args = parser.parse_args()

    warnings = []
    errors = []

    dataset_path = args.dataset
    out_path = args.out

    resolved_dataset = os.path.abspath(dataset_path)
    resolved_out = os.path.abspath(out_path)

    payload = {
        "dataset_path": dataset_path,
        "out_path": out_path,
        "classes_requested": [c.strip() for c in args.classes.split(",") if c.strip()],
        "classes_resolved": [],
        "class_mapping": {},
        "images_copied": 0,
        "label_files_written": 0,
        "annotation_lines_scanned": 0,
        "annotation_lines_extracted": 0,
        "warnings": warnings,
        "errors": errors
    }

    # Verify input path
    if not os.path.exists(resolved_dataset):
        errors.append(f"Dataset directory does not exist: {dataset_path}")
    elif not os.path.isdir(resolved_dataset):
        errors.append(f"Dataset path is not a directory: {dataset_path}")
    elif not check_yolo_layout(resolved_dataset):
        errors.append(f"Invalid YOLO dataset layout: {dataset_path}. Missing data.yaml/dataset.yaml, images/labels, or split subdirectories.")

    if resolved_dataset == resolved_out:
        errors.append("Dataset input and output paths must be different. Mutation in-place is not allowed.")

    if os.path.exists(resolved_out):
        if not args.force:
            errors.append(f"Output directory already exists: {out_path}. Use --force to overwrite.")
    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    # Read yaml
    yaml_file = None
    for f in os.listdir(resolved_dataset):
        if f.lower() in ["data.yaml", "dataset.yaml"]:
            yaml_file = f
            break

    class_map = {}
    yaml_data = {}
    if yaml_file:
        yaml_path = os.path.join(resolved_dataset, yaml_file)
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                import yaml
                yaml_data = yaml.safe_load(content)
            except Exception:
                yaml_data = parse_yaml_fallback(content)
                
            if yaml_data:
                names = yaml_data.get("names")
                if isinstance(names, list):
                    class_map = {idx: name for idx, name in enumerate(names)}
                elif isinstance(names, dict):
                    class_map = {int(k): v for k, v in names.items()}
        except Exception as e:
            warnings.append(f"Failed to read/parse YAML configuration: {e}")

    # Resolve target classes
    target_classes, resolve_errors = resolve_classes(args.classes, class_map)
    if resolve_errors:
        errors.extend(resolve_errors)
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    payload["classes_resolved"] = target_classes

    if os.path.exists(resolved_out) and args.force:
        try:
            if os.path.isdir(resolved_out):
                shutil.rmtree(resolved_out)
            else:
                os.remove(resolved_out)
        except Exception as e:
            errors.append(f"Failed to remove existing output path: {e}")
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Define remapping
    class_mapping = {}
    if args.no_remap:
        class_mapping = {cid: cid for cid in target_classes}
    else:
        class_mapping = {cid: idx for idx, cid in enumerate(target_classes)}
    
    payload["class_mapping"] = {str(k): v for k, v in class_mapping.items()}

    # Scan and process images and label files
    image_paths = find_all_images(resolved_dataset)
    
    images_copied_count = 0
    labels_written_count = 0
    lines_scanned_count = 0
    lines_extracted_count = 0

    # Ensure output directory exists before copying files
    os.makedirs(resolved_out, exist_ok=True)

    for img_path in image_paths:
        rel_img_path = os.path.relpath(img_path, resolved_dataset)
        rel_lbl_path, abs_lbl_path = find_label_for_image(resolved_dataset, rel_img_path)
        
        has_lbl = os.path.exists(abs_lbl_path)
        new_lines = []
        lbl_scanned = 0
        lbl_extracted = 0
        
        if has_lbl:
            new_lines, lbl_scanned, lbl_extracted = filter_and_edit_annotations(
                abs_lbl_path, target_classes, class_mapping
            )
            lines_scanned_count += lbl_scanned
            lines_extracted_count += lbl_extracted
            
        should_copy = (lbl_extracted > 0) or args.keep_empty_images
        
        if should_copy:
            # Copy image
            dst_img_path = os.path.join(resolved_out, rel_img_path)
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            try:
                shutil.copy2(img_path, dst_img_path)
                images_copied_count += 1
            except Exception as e:
                warnings.append(f"Failed to copy image {img_path} to {dst_img_path}: {e}")
                continue
                
            # Write label file (even if empty, to match image)
            dst_lbl_path = os.path.join(resolved_out, rel_lbl_path)
            os.makedirs(os.path.dirname(dst_lbl_path), exist_ok=True)
            try:
                with open(dst_lbl_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                labels_written_count += 1
            except Exception as e:
                warnings.append(f"Failed to write label file {dst_lbl_path}: {e}")

    # Generate output YAML file
    new_names = None
    if args.no_remap:
        # Keep original class map or list if no-remap
        if isinstance(yaml_data.get("names"), list):
            new_names = yaml_data.get("names")
        elif isinstance(yaml_data.get("names"), dict):
            new_names = yaml_data.get("names")
        else:
            new_names = {cid: class_map.get(cid, f"class_{cid}") for cid in target_classes}
    else:
        # Contiguous list in the order of target_classes
        new_names = [class_map.get(cid, f"class_{cid}") for cid in target_classes]

    new_nc = len(new_names) if isinstance(new_names, list) else len(target_classes)
    
    if yaml_file:
        dst_yaml_path = os.path.join(resolved_out, yaml_file)
        try:
            write_yolo_yaml(dst_yaml_path, yaml_data, new_nc, new_names)
        except Exception as e:
            warnings.append(f"Failed to write configuration file {dst_yaml_path}: {e}")
    else:
        # Create a default data.yaml
        dst_yaml_path = os.path.join(resolved_out, "data.yaml")
        default_yaml_data = {}
        # Try to infer train/val split paths if images were copied
        # e.g., if there's images/train, we write train: images/train
        for split in ["train", "val", "test"]:
            if os.path.exists(os.path.join(resolved_out, "images", split)):
                default_yaml_data[split] = f"images/{split}"
            elif os.path.exists(os.path.join(resolved_out, split, "images")):
                default_yaml_data[split] = f"{split}/images"
        
        try:
            write_yolo_yaml(dst_yaml_path, default_yaml_data, new_nc, new_names)
        except Exception as e:
            warnings.append(f"Failed to write default configuration file {dst_yaml_path}: {e}")

    if lines_extracted_count == 0:
        warnings.append("Zero annotation lines were extracted.")

    payload["images_copied"] = images_copied_count
    payload["label_files_written"] = labels_written_count
    payload["annotation_lines_scanned"] = lines_scanned_count
    payload["annotation_lines_extracted"] = lines_extracted_count

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab Dataset Category Extraction Report")
        print("=" * 70)
        print(f"Dataset path:       {payload['dataset_path']}")
        print(f"Output path:        {payload['out_path']}")
        print(f"Classes requested:  {', '.join(payload['classes_requested'])}")
        print(f"Classes resolved:   {', '.join(map(str, payload['classes_resolved']))}")
        print(f"Images copied:      {payload['images_copied']}")
        print(f"Labels written:     {payload['label_files_written']}")
        print(f"Lines scanned:      {payload['annotation_lines_scanned']}")
        print(f"Lines extracted:    {payload['annotation_lines_extracted']}")
        if payload["class_mapping"]:
            print("Class remapping (original -> new):")
            for k, v in sorted(payload["class_mapping"].items(), key=lambda x: int(x[0])):
                orig_name = class_map.get(int(k), f"class_{k}")
                new_name = new_names[v] if isinstance(new_names, list) else new_names.get(v, f"class_{v}")
                print(f"  - {k} ({orig_name}) -> {v} ({new_name})")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        print("=" * 70)

if __name__ == "__main__":
    main()
