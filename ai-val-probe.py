import sys
import os
import json
import argparse
import contextlib
import io
import logging

def main():
    parser = argparse.ArgumentParser(description="AI-first Validation Probe for YOLO-Lab")
    parser.add_argument("--model", type=str, help="Path to the model file")
    parser.add_argument("--data", type=str, help="Dataset config name or path (e.g. coco128.yaml)")
    parser.add_argument("--out", type=str, help="Output directory to save validation outputs")
    parser.add_argument("--imgsz", type=int, help="Inference image size (optional)")
    parser.add_argument("--conf", type=float, help="Confidence threshold (optional)")
    parser.add_argument("--split", type=str, help="Dataset split to validate on, e.g. val, test, train (optional)")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    args, unknown = parser.parse_known_args()

    warnings = []
    errors = []

    # Handle missing arguments in JSON mode cleanly
    if not args.model or not args.data or not args.out:
        missing_args = []
        if not args.model: missing_args.append("--model")
        if not args.data: missing_args.append("--data")
        if not args.out: missing_args.append("--out")
        
        err_msg = f"Missing required arguments: {', '.join(missing_args)}"
        if args.json:
            payload = {
                "model_path": args.model,
                "data_yaml": args.data,
                "out_dir": args.out,
                "task": None,
                "settings": {},
                "success": False,
                "metrics": {},
                "class_names": {},
                "generated_files": [],
                "warnings": [err_msg],
                "errors": [err_msg]
            }
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(err_msg, file=sys.stderr)
            parser.print_help()
            sys.exit(1)

    model_path = args.model
    data_yaml = args.data
    out_dir = args.out

    resolved_model_path = os.path.abspath(model_path)
    model_exists = os.path.exists(resolved_model_path)

    if not model_exists:
        errors.append(f"Model file does not exist: {model_path}")

    if not data_yaml.strip():
        errors.append("Dataset name/path is empty")

    payload = {
        "model_path": model_path,
        "data_yaml": data_yaml,
        "out_dir": out_dir,
        "task": None,
        "settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "split": args.split
        },
        "success": False,
        "metrics": {},
        "class_names": {},
        "generated_files": [],
        "warnings": warnings,
        "errors": errors
    }

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)

    ultralytics_available = False
    try:
        import ultralytics
        ultralytics_available = True
    except Exception as e:
        err_msg = f"Failed to import ultralytics: {e}"
        errors.append(err_msg)
        payload["errors"] = errors
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(err_msg, file=sys.stderr)
            sys.exit(1)

    try:
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
    except Exception:
        pass

    f_stdout = io.StringIO()
    f_stderr = io.StringIO()

    try:
        with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(resolved_model_path)
            payload["task"] = getattr(model, "task", None)
            
            # Extract class names
            names = getattr(model, "names", {})
            if isinstance(names, dict):
                payload["class_names"] = {str(k): v for k, v in names.items()}
            else:
                payload["class_names"] = names

            # Setup output directories mapped to YOLO project and name parameters
            resolved_out = os.path.abspath(out_dir)
            project_dir = os.path.dirname(resolved_out)
            name_dir = os.path.basename(resolved_out)

            # Build validation parameters
            val_kwargs = {
                "data": data_yaml,
                "project": project_dir,
                "name": name_dir,
                "save": True,
                "plots": True,
                "verbose": False
            }
            if args.conf is not None:
                val_kwargs["conf"] = args.conf
            if args.imgsz is not None:
                val_kwargs["imgsz"] = args.imgsz
            if args.split is not None:
                val_kwargs["split"] = args.split

            # Run validation
            metrics = model.val(**val_kwargs)

        payload["success"] = True

        # Extract metrics safely
        box_metrics = {}
        if hasattr(metrics, "box") and metrics.box is not None:
            bm = metrics.box
            box_metrics = {
                "map50": getattr(bm, "map50", None),
                "map50_95": getattr(bm, "map", None),
                "mp": getattr(bm, "mp", None),
                "mr": getattr(bm, "mr", None),
            }

        seg_metrics = {}
        if hasattr(metrics, "seg") and metrics.seg is not None:
            sm = metrics.seg
            seg_metrics = {
                "map50": getattr(sm, "map50", None),
                "map50_95": getattr(sm, "map", None),
                "mp": getattr(sm, "mp", None),
                "mr": getattr(sm, "mr", None),
            }

        speed = getattr(metrics, "speed", {})
        fitness = getattr(metrics, "fitness", None)

        payload["metrics"] = {
            "box": box_metrics,
            "seg": seg_metrics,
            "speed_ms": speed,
            "fitness": fitness,
            "results_dict": getattr(metrics, "results_dict", {})
        }

        # Track generated output files
        generated_files = []
        val_out_dir = os.path.join(project_dir, name_dir)
        if os.path.exists(val_out_dir):
            for root, dirs, files in os.walk(val_out_dir):
                for file in files:
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, os.getcwd())
                    generated_files.append(rel_p)
        payload["generated_files"] = generated_files

    except Exception as e:
        err_msg = f"Failed during validation: {e}"
        errors.append(err_msg)
        payload["errors"] = errors
        
        captured_stdout = f_stdout.getvalue().strip()
        captured_stderr = f_stderr.getvalue().strip()
        if captured_stdout:
            warnings.append(f"Captured stdout: {captured_stdout}")
        if captured_stderr:
            warnings.append(f"Captured stderr: {captured_stderr}")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab Validation Probe Summary")
        print("=" * 70)
        print(f"Model Path:      {payload['model_path']}")
        print(f"Dataset YAML:    {payload['data_yaml']}")
        print(f"Output Dir:      {payload['out_dir']}")
        print(f"Task:            {payload['task']}")
        print(f"Success:         {payload['success']}")
        
        if payload['success']:
            m = payload["metrics"]
            if m["box"]:
                print("\nDetection Metrics:")
                print(f"  mAP50:         {m['box']['map50']:.4f}" if m['box']['map50'] is not None else "  mAP50: N/A")
                print(f"  mAP50-95:      {m['box']['map50_95']:.4f}" if m['box']['map50_95'] is not None else "  mAP50-95: N/A")
                print(f"  Precision (m): {m['box']['mp']:.4f}" if m['box']['mp'] is not None else "  Precision: N/A")
                print(f"  Recall (m):    {m['box']['mr']:.4f}" if m['box']['mr'] is not None else "  Recall: N/A")
            if m["seg"]:
                print("\nSegmentation Metrics:")
                print(f"  mAP50 (mask):  {m['seg']['map50']:.4f}" if m['seg']['map50'] is not None else "  mAP50 (mask): N/A")
                print(f"  mAP50-95(msk): {m['seg']['map50_95']:.4f}" if m['seg']['map50_95'] is not None else "  mAP50-95(msk): N/A")
            if m["fitness"] is not None:
                print(f"Fitness Score:   {m['fitness']:.4f}")
            if m["speed_ms"]:
                print("\nInference Speed (ms/image):")
                for k, v in m["speed_ms"].items():
                    print(f"  {k}: {v:.2f}ms")
            
            print(f"\nGenerated Files: {len(payload['generated_files'])} items saved.")
            if len(payload['generated_files']) > 0:
                print("  Examples:")
                for f in payload['generated_files'][:5]:
                    print(f"    - {f}")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  - {e}")
        print("=" * 70)

if __name__ == "__main__":
    main()
