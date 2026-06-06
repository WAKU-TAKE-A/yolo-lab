import sys
import os
import json
import argparse
import contextlib
import io
import logging
import shutil

def main():
    parser = argparse.ArgumentParser(description="Export standard models to ONNX for YOLO-Lab")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")
    args, unknown = parser.parse_known_args()

    warnings = []
    errors = []

    # Import ultralytics
    ultralytics_available = False
    try:
        import ultralytics
        ultralytics_available = True
    except Exception as e:
        err_msg = f"Failed to import ultralytics: {e}"
        errors.append(err_msg)
        if args.json:
            print(json.dumps({"exports": [], "warnings": warnings, "errors": errors}, indent=2))
            sys.exit(0)
        else:
            print(err_msg, file=sys.stderr)
            sys.exit(1)

    try:
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
    except Exception:
        pass

    # Ensure output directory exists
    onnx_dir = os.path.join("models", "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    configs = [
        {"src": "models/standard/yolov8s.pt", "name": "yolov8s.onnx", "kwargs": {}},
        {"src": "models/standard/yolov8s-seg.pt", "name": "yolov8s-seg.onnx", "kwargs": {}},
        {"src": "models/standard/yolo26s.pt", "name": "yolo26s.onnx", "kwargs": {}},
        {"src": "models/standard/yolo26s.pt", "name": "yolo26s_no_end2end.onnx", "kwargs": {"end2end": False}},
        {"src": "models/standard/yolo26s-seg.pt", "name": "yolo26s-seg.onnx", "kwargs": {}},
        {"src": "models/standard/yolo26s-seg.pt", "name": "yolo26s-seg_no_end2end.onnx", "kwargs": {"end2end": False}}
    ]

    export_results = []

    for cfg in configs:
        src_path = cfg["src"]
        dest_filename = cfg["name"]
        dest_path = os.path.join(onnx_dir, dest_filename)
        resolved_src = os.path.abspath(src_path)

        res = {
            "source_model_path": src_path,
            "exported_onnx_path": None,
            "task": None,
            "settings": {
                "format": "onnx",
                "imgsz": 640,
                "opset": 17,
                **cfg["kwargs"]
            },
            "success": False,
            "file_size_bytes": None,
            "warnings": [],
            "errors": []
        }

        if not os.path.exists(resolved_src):
            err_msg = f"Source model does not exist: {src_path}"
            res["errors"].append(err_msg)
            warnings.append(f"Skipped {dest_filename} because source was missing")
            export_results.append(res)
            continue

        f_stdout = io.StringIO()
        f_stderr = io.StringIO()

        try:
            with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
                from ultralytics import YOLO
                model = YOLO(resolved_src)
                res["task"] = getattr(model, "task", None)

                # Configure export arguments
                export_args = {
                    "format": "onnx",
                    "imgsz": 640,
                    "opset": 17,
                    "verbose": False
                }
                export_args.update(cfg["kwargs"])

                # Run export
                temp_onnx_path = model.export(**export_args)

            # Move file to models/onnx/
            if temp_onnx_path and os.path.exists(temp_onnx_path):
                shutil.move(temp_onnx_path, dest_path)
                res["success"] = True
                res["exported_onnx_path"] = os.path.abspath(dest_path)
                res["file_size_bytes"] = os.path.getsize(dest_path)
            else:
                # Fallback: check if the file was created at the default location
                # usually replacing .pt with .onnx in the same folder
                expected_temp_path = resolved_src.replace(".pt", ".onnx")
                if os.path.exists(expected_temp_path):
                    shutil.move(expected_temp_path, dest_path)
                    res["success"] = True
                    res["exported_onnx_path"] = os.path.abspath(dest_path)
                    res["file_size_bytes"] = os.path.getsize(dest_path)
                else:
                    # Also check if it might be in CWD
                    cwd_expected_path = os.path.join(os.getcwd(), os.path.basename(resolved_src).replace(".pt", ".onnx"))
                    if os.path.exists(cwd_expected_path):
                        shutil.move(cwd_expected_path, dest_path)
                        res["success"] = True
                        res["exported_onnx_path"] = os.path.abspath(dest_path)
                        res["file_size_bytes"] = os.path.getsize(dest_path)
                    else:
                        raise FileNotFoundError(f"Export did not produce ONNX file at expected path: {expected_temp_path}")

        except Exception as e:
            res["errors"].append(str(e))
            captured_stdout = f_stdout.getvalue().strip()
            captured_stderr = f_stderr.getvalue().strip()
            if captured_stdout:
                res["warnings"].append(f"Captured stdout: {captured_stdout}")
            if captured_stderr:
                res["warnings"].append(f"Captured stderr: {captured_stderr}")

        export_results.append(res)

    payload = {
        "exports": export_results,
        "warnings": warnings,
        "errors": errors
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab ONNX Export Summary")
        print("=" * 70)
        for r in export_results:
            status = "Succeeded" if r["success"] else "Failed"
            print(f"Source: {r['source_model_path']}")
            print(f"  ONNX:  {r['exported_onnx_path']} ({status})")
            if r["file_size_bytes"]:
                print(f"  Size:  {r['file_size_bytes']} bytes")
            if r["errors"]:
                print(f"  Errors: {r['errors']}")
            print("-" * 50)
        print("=" * 70)

if __name__ == "__main__":
    main()
