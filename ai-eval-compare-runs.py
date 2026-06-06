import os
import sys
import json
import argparse
import csv


def confidence_summary(values):
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    payload = {
        "before_run": args.before,
        "after_run": args.after,
        "before_model_path": None,
        "after_model_path": None,
        "image_count_before": 0,
        "image_count_after": 0,
        "detection_count_before": 0,
        "detection_count_after": 0,
        "class_counts_before": {},
        "class_counts_after": {},
        "confidence_summary_before": None,
        "confidence_summary_after": None,
        "review_records_before": 0,
        "review_records_after": 0,
        "warnings": [],
        "errors": []
    }
    
    def die():
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for e in payload["errors"]: print(e, file=sys.stderr)
        sys.exit(1 if payload["errors"] else 0)

    if not os.path.isdir(args.before):
        payload["errors"].append(f"Before run directory does not exist: {args.before}")
        
    if not os.path.isdir(args.after):
        payload["errors"].append(f"After run directory does not exist: {args.after}")
        
    if payload["errors"]:
        die()
        
    def parse_run(run_dir):
        m_path = os.path.join(run_dir, "manifest.json")
        r_path = os.path.join(run_dir, "results.csv")
        rev_path = os.path.join(run_dir, "review.jsonl")
        
        if not os.path.exists(m_path):
            return None, f"manifest.json missing in {run_dir}"
            
        try:
            with open(m_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return None, f"Failed to read manifest.json: {e}"
            
        model_path = manifest.get("model")
        img_count = manifest.get("image_count", 0)
        
        det_count = 0
        cls_counts = {}
        conf_values = []
        if os.path.exists(r_path):
            try:
                with open(r_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames is None:
                        return None, f"results.csv has no header in {run_dir}"
                    required_cols = {"class_name", "confidence"}
                    missing_cols = sorted(required_cols - set(reader.fieldnames))
                    if missing_cols:
                        return None, f"results.csv missing columns in {run_dir}: {', '.join(missing_cols)}"
                    for row in reader:
                        det_count += 1
                        cname = row.get("class_name") or ""
                        cls_counts[cname] = cls_counts.get(cname, 0) + 1
                        try:
                            conf_values.append(float(row["confidence"]))
                        except (TypeError, ValueError):
                            pass
            except Exception as e:
                return None, f"Failed to read results.csv: {e}"
        else:
            return None, f"results.csv missing in {run_dir}"
                
        rev_count = 0
        if os.path.exists(rev_path):
            try:
                with open(rev_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip(): rev_count += 1
            except: pass
            
        return {
            "model_path": model_path,
            "image_count": img_count,
            "detection_count": det_count,
            "class_counts": cls_counts,
            "confidence_summary": confidence_summary(conf_values),
            "review_records": rev_count
        }, None

    b_data, b_err = parse_run(args.before)
    if b_err: payload["errors"].append(b_err)
    else:
        payload["before_model_path"] = b_data["model_path"]
        payload["image_count_before"] = b_data["image_count"]
        payload["detection_count_before"] = b_data["detection_count"]
        payload["class_counts_before"] = b_data["class_counts"]
        payload["confidence_summary_before"] = b_data["confidence_summary"]
        payload["review_records_before"] = b_data["review_records"]

    a_data, a_err = parse_run(args.after)
    if a_err: payload["errors"].append(a_err)
    else:
        payload["after_model_path"] = a_data["model_path"]
        payload["image_count_after"] = a_data["image_count"]
        payload["detection_count_after"] = a_data["detection_count"]
        payload["class_counts_after"] = a_data["class_counts"]
        payload["confidence_summary_after"] = a_data["confidence_summary"]
        payload["review_records_after"] = a_data["review_records"]

    if payload["errors"]:
        die()
        
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Comparison: {args.before} vs {args.after}")
        print(f"Images: {payload['image_count_before']} vs {payload['image_count_after']}")
        print(f"Detections: {payload['detection_count_before']} vs {payload['detection_count_after']}")

if __name__ == "__main__":
    main()
