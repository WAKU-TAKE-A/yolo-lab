import os
import sys
import json
import argparse
import shutil
from datetime import datetime, timezone, timedelta

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

def find_label_files(dataset_dir):
    has_labels_folder = False
    for root, dirs, files in os.walk(dataset_dir):
        if "labels" in [d.lower() for d in dirs]:
            has_labels_folder = True
            break
            
    label_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        rel_dir = os.path.relpath(root, dataset_dir)
        parts = rel_dir.split(os.sep)
        is_labels_path = "labels" in [p.lower() for p in parts]
        
        if has_labels_folder and not is_labels_path:
            continue
            
        for f in files:
            if f.lower().endswith(".txt"):
                label_paths.append(os.path.join(root, f))
    return label_paths

def filter_label_files(label_paths, dataset_dir, selected_images=None, selected_labels=None):
    image_stems = set()
    if selected_images:
        for img in selected_images:
            stem = os.path.splitext(os.path.basename(img))[0]
            image_stems.add(stem)
            
    label_stems = set()
    if selected_labels:
        for lbl in selected_labels:
            stem = os.path.splitext(os.path.basename(lbl))[0]
            label_stems.add(stem)
            
    filtered_paths = []
    warnings_list = []
    
    matched_image_stems = set()
    matched_label_stems = set()
    
    for path in label_paths:
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]
        
        keep = True
        if selected_images and selected_labels:
            match_img = stem in image_stems
            match_lbl = stem in label_stems
            if match_img:
                matched_image_stems.add(stem)
            if match_lbl:
                matched_label_stems.add(stem)
            keep = match_img or match_lbl
        elif selected_images:
            match_img = stem in image_stems
            if match_img:
                matched_image_stems.add(stem)
            keep = match_img
        elif selected_labels:
            match_lbl = stem in label_stems
            if match_lbl:
                matched_label_stems.add(stem)
            keep = match_lbl
            
        if keep:
            filtered_paths.append(path)
            
    if selected_images:
        unmatched_imgs = image_stems - matched_image_stems
        for ui in unmatched_imgs:
            warnings_list.append(f"No label file found matching image stem: {ui}")
            
    if selected_labels:
        unmatched_lbls = label_stems - matched_label_stems
        for ul in unmatched_lbls:
            warnings_list.append(f"No label file found matching label stem: {ul}")
            
    return filtered_paths, warnings_list

def edit_label_line(line, from_class, to_class):
    stripped = line.rstrip('\r\n')
    if not stripped:
        return line, False
        
    if stripped.startswith('#') or stripped.startswith('//'):
        return line, False
        
    tokens = stripped.split()
    if not tokens:
        return line, False
        
    first_token = tokens[0]
    try:
        class_id = int(first_token)
    except ValueError:
        return line, False
        
    modified = False
    if from_class is None or class_id == from_class:
        tokens[0] = str(to_class)
        modified = True
        
    if modified:
        suffix = line[len(stripped):]
        new_line = " ".join(tokens) + suffix
        return new_line, True
    else:
        return line, False

def main():
    parser = argparse.ArgumentParser(description="AI-first YOLO class label edit utility")
    parser.add_argument("--dataset", type=str, required=True, help="Path to source YOLO dataset")
    parser.add_argument("--out", type=str, required=True, help="Path to output dataset directory")
    parser.add_argument("--from-class", type=str, help="Optional class ID to match (if omitted, all lines are edited)")
    parser.add_argument("--to-class", type=str, required=True, help="New class ID to write")
    parser.add_argument("--images", type=str, help="Optional comma-separated image stems or filenames")
    parser.add_argument("--labels", type=str, help="Optional comma-separated label stems or filenames")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing output directory")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    args = parser.parse_args()

    warnings = []
    errors = []

    dataset_path = args.dataset
    out_path = args.out

    resolved_dataset = os.path.abspath(dataset_path)
    resolved_out = os.path.abspath(out_path)

    # Class ID validations
    from_class = None
    if args.from_class is not None:
        try:
            from_class = int(args.from_class)
            if from_class < 0:
                errors.append(f"Invalid --from-class ID (must be non-negative): {args.from_class}")
        except ValueError:
            errors.append(f"Invalid --from-class ID (must be an integer): {args.from_class}")
            
    to_class = None
    try:
        to_class = int(args.to_class)
        if to_class < 0:
            errors.append(f"Invalid --to-class ID (must be non-negative): {args.to_class}")
    except ValueError:
        errors.append(f"Invalid --to-class ID (must be an integer): {args.to_class}")

    mode = "all_labels"
    selected_images = None
    selected_labels = None

    if args.images:
        selected_images = [s.strip() for s in args.images.split(",") if s.strip()]
        mode = "selected_images"
    if args.labels:
        selected_labels = [s.strip() for s in args.labels.split(",") if s.strip()]
        if mode == "selected_images":
            mode = "selected_images_and_labels"
        else:
            mode = "selected_labels"

    payload = {
        "dataset_path": dataset_path,
        "out_path": out_path,
        "mode": mode,
        "from_class": from_class,
        "to_class": to_class,
        "label_files_scanned": 0,
        "label_files_modified": 0,
        "annotation_lines_scanned": 0,
        "annotation_lines_modified": 0,
        "changed_files": [],
        "warnings": warnings,
        "errors": errors
    }

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    # Resolve paths validation
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
        else:
            try:
                if os.path.isdir(resolved_out):
                    shutil.rmtree(resolved_out)
                else:
                    os.remove(resolved_out)
            except Exception as e:
                errors.append(f"Failed to remove existing output path: {e}")

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    # 1. Copy the dataset
    try:
        shutil.copytree(resolved_dataset, resolved_out)
    except Exception as e:
        errors.append(f"Failed to copy dataset directory: {e}")
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(f"Failed to copy dataset directory: {e}", file=sys.stderr)
            sys.exit(1)

    # 2. Find and filter label files in the copied directory
    copied_label_paths = find_label_files(resolved_out)
    filtered_copied_label_paths, filter_warnings = filter_label_files(
        copied_label_paths, resolved_out, selected_images, selected_labels
    )
    warnings.extend(filter_warnings)

    files_scanned_count = 0
    files_modified_count = 0
    lines_scanned_count = 0
    lines_modified_count = 0
    changed_files = []

    for path in filtered_copied_label_paths:
        files_scanned_count += 1
        modified_lines_in_file = 0
        new_lines = []
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                    tokens = stripped.split()
                    if tokens:
                        try:
                            int(tokens[0])
                            lines_scanned_count += 1
                        except ValueError:
                            pass
                
                new_line, is_mod = edit_label_line(line, from_class, to_class)
                new_lines.append(new_line)
                if is_mod:
                    lines_modified_count += 1
                    modified_lines_in_file += 1
                    
            if modified_lines_in_file > 0:
                files_modified_count += 1
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                
                rel_path = os.path.relpath(path, resolved_out)
                changed_files.append({
                    "file": rel_path,
                    "lines_modified": modified_lines_in_file
                })
        except Exception as e:
            warnings.append(f"Failed to process label file {path}: {e}")

    # Zero lines modified warning
    if lines_modified_count == 0:
        warnings.append("Zero annotation lines were modified.")

    # Populate final payload values
    payload["label_files_scanned"] = files_scanned_count
    payload["label_files_modified"] = files_modified_count
    payload["annotation_lines_scanned"] = lines_scanned_count
    payload["annotation_lines_modified"] = lines_modified_count
    payload["changed_files"] = changed_files

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab Dataset Class Edit Report")
        print("=" * 70)
        print(f"Dataset path:      {payload['dataset_path']}")
        print(f"Output path:       {payload['out_path']}")
        print(f"Operation Mode:    {payload['mode']}")
        print(f"From class:        {payload['from_class'] if payload['from_class'] is not None else 'ALL'}")
        print(f"To class:          {payload['to_class']}")
        print(f"Files scanned:     {payload['label_files_scanned']}")
        print(f"Files modified:    {payload['label_files_modified']}")
        print(f"Lines scanned:     {payload['annotation_lines_scanned']}")
        print(f"Lines modified:    {payload['annotation_lines_modified']}")
        if payload["changed_files"]:
            print("Changed files:")
            for cf in payload["changed_files"]:
                print(f"  - {cf['file']}: {cf['lines_modified']} lines")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        print("=" * 70)

if __name__ == "__main__":
    main()
