import os
import sys
import json
import csv
import argparse

def count_file_lines(path):
    if not os.path.exists(path):
        return 0
    try:
        cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                cnt += 1
        return cnt
    except Exception:
        return 0

def count_csv_rows(path):
    if not os.path.exists(path):
        return 0
    try:
        cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # skip header
            next(reader, None)
            for _ in reader:
                cnt += 1
        return cnt
    except Exception:
        return 0

def parse_yaml_fallback(file_content):
    import re
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

    for key in ['train', 'val', 'test']:
        m = re.search(rf'^{key}:\s*(.*)', file_content, re.MULTILINE)
        if m:
            data[key] = m.group(1).strip().strip("'\"")
    return data

def count_yolo_files(base_dir, data_yaml_content=None):
    train_path = val_path = test_path = None
    if data_yaml_content:
        train_path = data_yaml_content.get("train")
        val_path = data_yaml_content.get("val")
        test_path = data_yaml_content.get("test")
    
    counts = {
        "train": {"images": 0, "labels": 0},
        "val": {"images": 0, "labels": 0},
        "test": {"images": 0, "labels": 0}
    }
    
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    def count_files(dir_path, exts=None):
        if not dir_path or not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            return 0
        try:
            cnt = 0
            for f in os.listdir(dir_path):
                if os.path.isfile(os.path.join(dir_path, f)):
                    if exts:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in exts:
                            cnt += 1
                    else:
                        cnt += 1
            return cnt
        except Exception:
            return 0

    # Pattern 1: base_dir/images/split and base_dir/labels/split
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base_dir, "images", split)
        lbl_dir = os.path.join(base_dir, "labels", split)
        counts[split]["images"] += count_files(img_dir, image_exts)
        counts[split]["labels"] += count_files(lbl_dir, {".txt"})

    # Pattern 2: base_dir/split/images and base_dir/split/labels
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")
        counts[split]["images"] += count_files(img_dir, image_exts)
        counts[split]["labels"] += count_files(lbl_dir, {".txt"})

    # Pattern 3: using paths from yaml
    for split, yaml_val in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if yaml_val:
            resolved_path = os.path.normpath(os.path.join(base_dir, yaml_val))
            if os.path.exists(resolved_path) and os.path.isdir(resolved_path):
                img_cnt = count_files(resolved_path, image_exts)
                if img_cnt > 0:
                    counts[split]["images"] = max(counts[split]["images"], img_cnt)
                
                lbl_resolved = None
                if "images" in resolved_path:
                    lbl_resolved = resolved_path.replace("images", "labels")
                elif "images" in yaml_val:
                    lbl_resolved = os.path.normpath(os.path.join(base_dir, yaml_val.replace("images", "labels")))
                
                if lbl_resolved and os.path.exists(lbl_resolved) and os.path.isdir(lbl_resolved):
                    lbl_cnt = count_files(lbl_resolved, {".txt"})
                    counts[split]["labels"] = max(counts[split]["labels"], lbl_cnt)
    
    return counts

def main():
    parser = argparse.ArgumentParser(description="Dataset probe utility for YOLO-Lab")
    parser.add_argument("--path", type=str, help="Path to dataset file or directory")
    parser.add_argument("--dataset", type=str, help="Alias for --path")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    args = parser.parse_args()
    
    input_path = args.path or args.dataset
    if not input_path:
        parser.error("Either --path or --dataset is required to specify the input path.")

    warnings = []
    errors = []

    resolved_path = os.path.abspath(input_path)

    payload = {
        "input_path": input_path,
        "resolved_path": resolved_path,
        "detected_types": [],
        "files_found": [],
        "warnings": warnings,
        "errors": errors
    }

    if not os.path.exists(resolved_path):
        errors.append(f"Path does not exist: {input_path}")

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    is_dir = os.path.isdir(resolved_path)

    coco_details = []
    ls_details = []
    yolo_details = None
    eval_run_details = None
    eval_candidates_details = None

    if is_dir:
        # 1. Check AI Eval Run
        manifest_path = os.path.join(resolved_path, "manifest.json")
        results_csv = os.path.join(resolved_path, "results.csv")
        if os.path.exists(manifest_path) and os.path.exists(results_csv):
            payload["detected_types"].append("ai_eval_run")
            eval_run_details = {
                "manifest": None,
                "results_row_count": 0,
                "review_line_count": 0,
                "subfolders": []
            }
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    eval_run_details["manifest"] = json.load(f)
            except Exception as e:
                warnings.append(f"Failed to read manifest.json: {e}")
            
            eval_run_details["results_row_count"] = count_csv_rows(results_csv)
            review_jsonl = os.path.join(resolved_path, "review.jsonl")
            if os.path.exists(review_jsonl):
                eval_run_details["review_line_count"] = count_file_lines(review_jsonl)
            
            for folder in ["images", "overlays", "predictions"]:
                if os.path.exists(os.path.join(resolved_path, folder)):
                    eval_run_details["subfolders"].append(folder)

        # 2. Check AI Eval Candidates
        candidates_json = os.path.join(resolved_path, "candidates.json")
        candidates_csv = os.path.join(resolved_path, "candidates.csv")
        if os.path.exists(candidates_json) and os.path.exists(candidates_csv):
            payload["detected_types"].append("ai_eval_candidates")
            eval_candidates_details = {
                "candidate_count": 0,
                "selected_statuses": [],
                "images_txt_count": 0,
                "status_counts": {}
            }
            try:
                with open(candidates_json, "r", encoding="utf-8") as f:
                    cand_data = json.load(f)
                    eval_candidates_details["candidate_count"] = cand_data.get("candidate_count", 0)
                    eval_candidates_details["selected_statuses"] = cand_data.get("selected_statuses", [])
                    
                    # Compute status counts
                    status_counts = {}
                    for c in cand_data.get("candidates", []):
                        st = c.get("status", "unknown")
                        status_counts[st] = status_counts.get(st, 0) + 1
                    eval_candidates_details["status_counts"] = status_counts
            except Exception as e:
                warnings.append(f"Failed to read candidates.json: {e}")
            
            images_txt = os.path.join(resolved_path, "images.txt")
            if os.path.exists(images_txt):
                eval_candidates_details["images_txt_count"] = count_file_lines(images_txt)

        # 3. Check YOLO Dataset
        data_yaml = None
        for f in os.listdir(resolved_path):
            if f.lower() in ["data.yaml", "dataset.yaml"]:
                data_yaml = os.path.join(resolved_path, f)
                break
        
        if data_yaml:
            payload["detected_types"].append("yolo_dataset")
            yolo_details = {
                "yaml_file": os.path.basename(data_yaml),
                "train_val_test_counts": {},
                "class_map": {}
            }
            yaml_data = {}
            try:
                with open(data_yaml, "r", encoding="utf-8") as f:
                    content = f.read()
                
                try:
                    import yaml
                    yaml_data = yaml.safe_load(content)
                except Exception:
                    yaml_data = parse_yaml_fallback(content)
                
                if yaml_data:
                    names = yaml_data.get("names")
                    if isinstance(names, list):
                        yolo_details["class_map"] = {idx: name for idx, name in enumerate(names)}
                    elif isinstance(names, dict):
                        yolo_details["class_map"] = {int(k): v for k, v in names.items()}
            except Exception as e:
                warnings.append(f"Failed to read/parse YAML file {data_yaml}: {e}")
            
            yolo_details["train_val_test_counts"] = count_yolo_files(resolved_path, yaml_data)

        # 4. Check for COCO JSON inside folders
        json_files = []
        try:
            for f in os.listdir(resolved_path):
                if f.lower().endswith(".json"):
                    json_files.append(os.path.join(resolved_path, f))
            ann_dir = os.path.join(resolved_path, "annotations")
            if os.path.exists(ann_dir) and os.path.isdir(ann_dir):
                for f in os.listdir(ann_dir):
                    if f.lower().endswith(".json"):
                        json_files.append(os.path.join(ann_dir, f))
        except Exception as e:
            warnings.append(f"Failed to scan JSON files in directory: {e}")

        for jf in json_files:
            # check if it's COCO or Label Studio
            try:
                # To prevent loading huge files completely if not necessary, we load normally
                # but if they are too big we just catch MemoryError/errors.
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check COCO keys
                if isinstance(data, dict) and "images" in data and "annotations" in data and "categories" in data:
                    payload["detected_types"].append("coco_json")
                    coco_details.append({
                        "file": os.path.relpath(jf, resolved_path),
                        "image_count": len(data.get("images", [])),
                        "annotation_count": len(data.get("annotations", [])),
                        "categories": [c.get("name") for c in data.get("categories", []) if c.get("name")]
                    })
                
                # Check Label Studio
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "id" in data[0] and "data" in data[0] and ("annotations" in data[0] or "predictions" in data[0]):
                    payload["detected_types"].append("label_studio_export")
                    
                    labels = set()
                    for task in data[:100]:
                        for ann in task.get("annotations", []):
                            for res in ann.get("result", []):
                                val = res.get("value", {})
                                for key in ["labels", "rectanglelabels", "polygonlabels", "choices"]:
                                    if key in val and isinstance(val[key], list):
                                        labels.update(val[key])
                    
                    ls_details.append({
                        "file": os.path.relpath(jf, resolved_path),
                        "task_count": len(data),
                        "approximate_labels": sorted(list(labels))
                    })
            except Exception:
                pass # Not a COCO or LS JSON, or failed to parse

    else:
        # File path check (COCO JSON or LS export)
        if resolved_path.lower().endswith(".json"):
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check COCO keys
                if isinstance(data, dict) and "images" in data and "annotations" in data and "categories" in data:
                    payload["detected_types"].append("coco_json")
                    coco_details.append({
                        "file": os.path.basename(resolved_path),
                        "image_count": len(data.get("images", [])),
                        "annotation_count": len(data.get("annotations", [])),
                        "categories": [c.get("name") for c in data.get("categories", []) if c.get("name")]
                    })
                
                # Check Label Studio
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "id" in data[0] and "data" in data[0] and ("annotations" in data[0] or "predictions" in data[0]):
                    payload["detected_types"].append("label_studio_export")
                    
                    labels = set()
                    for task in data[:100]:
                        for ann in task.get("annotations", []):
                            for res in ann.get("result", []):
                                val = res.get("value", {})
                                for key in ["labels", "rectanglelabels", "polygonlabels", "choices"]:
                                    if key in val and isinstance(val[key], list):
                                        labels.update(val[key])
                    
                    ls_details.append({
                        "file": os.path.basename(resolved_path),
                        "task_count": len(data),
                        "approximate_labels": sorted(list(labels))
                    })
            except Exception as e:
                warnings.append(f"Failed to read/parse JSON file: {e}")

    # Deduplicate detected types
    payload["detected_types"] = sorted(list(set(payload["detected_types"])))
    if not payload["detected_types"]:
        payload["detected_types"].append("unknown")

    # Add detail blocks to payload if populated
    if coco_details:
        payload["coco_details"] = coco_details
    if ls_details:
        payload["label_studio_details"] = ls_details
    if yolo_details:
        payload["yolo_details"] = yolo_details
    if eval_run_details:
        payload["eval_run_details"] = eval_run_details
    if eval_candidates_details:
        payload["eval_candidates_details"] = eval_candidates_details

    # List key files found
    try:
        if is_dir:
            for item in os.listdir(resolved_path):
                if os.path.isfile(os.path.join(resolved_path, item)):
                    payload["files_found"].append(item)
    except Exception:
        pass

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab Dataset Probe Report")
        print("=" * 70)
        print(f"Path inspected:  {payload['input_path']}")
        print(f"Detected types:  {', '.join(payload['detected_types'])}")
        
        if eval_run_details:
            print("\n--- AI Eval Run Details ---")
            m = eval_run_details["manifest"] or {}
            print(f"  Eval ID:           {m.get('eval_id', 'N/A')}")
            print(f"  Model used:        {m.get('model', 'N/A')}")
            print(f"  Images count:      {eval_run_details['results_row_count']}")
            print(f"  Detections count:  {m.get('detection_count', 'N/A')}")
            print(f"  Review records:    {eval_run_details['review_line_count']}")
            print(f"  Subfolders found:  {', '.join(eval_run_details['subfolders'])}")

        if eval_candidates_details:
            print("\n--- AI Eval Candidates Details ---")
            print(f"  Candidate count:   {eval_candidates_details['candidate_count']}")
            print(f"  Filtered statuses: {', '.join(eval_candidates_details['selected_statuses'])}")
            print(f"  Images txt count:  {eval_candidates_details['images_txt_count']}")
            if eval_candidates_details["status_counts"]:
                print("  Status counts:")
                for k, v in sorted(eval_candidates_details["status_counts"].items()):
                    print(f"    - {k}: {v}")

        if yolo_details:
            print("\n--- YOLO Dataset Details ---")
            print(f"  YAML file:         {yolo_details['yaml_file']}")
            print("  Split counts (images / labels):")
            for split, counts in sorted(yolo_details["train_val_test_counts"].items()):
                print(f"    - {split}: {counts['images']} images, {counts['labels']} labels")
            if yolo_details["class_map"]:
                print(f"  Classes ({len(yolo_details['class_map'])} found):")
                for k, v in sorted(yolo_details["class_map"].items())[:5]:
                    print(f"    - {k}: {v}")
                if len(yolo_details["class_map"]) > 5:
                    print(f"    ... and {len(yolo_details['class_map']) - 5} more")

        if coco_details:
            print("\n--- COCO JSON Details ---")
            for idx, c in enumerate(coco_details):
                print(f"  File:              {c['file']}")
                print(f"    Images count:      {c['image_count']}")
                print(f"    Annotations count: {c['annotation_count']}")
                print(f"    Categories ({len(c['categories'])}): {', '.join(c['categories'][:5])}" + ("..." if len(c['categories']) > 5 else ""))

        if ls_details:
            print("\n--- Label Studio Export Details ---")
            for idx, ls in enumerate(ls_details):
                print(f"  File:              {ls['file']}")
                print(f"    Tasks count:       {ls['task_count']}")
                print(f"    Labels found:      {', '.join(ls['approximate_labels'][:5])}" + ("..." if len(ls['approximate_labels']) > 5 else ""))

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        print("=" * 70)

if __name__ == "__main__":
    main()
