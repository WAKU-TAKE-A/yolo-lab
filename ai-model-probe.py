import sys
import os
import json
import argparse
import contextlib
import io
import logging

def main():
    parser = argparse.ArgumentParser(description="AI-first Model Probe for YOLO-Lab")
    parser.add_argument("--model", type=str, help="Path to the model file to inspect")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")
    
    args, unknown = parser.parse_known_args()
    
    warnings = []
    errors = []
    
    if not args.model:
        if args.json:
            payload = {
                "model": {
                    "path": None,
                    "resolved_path": None,
                    "exists": False,
                    "size_bytes": None,
                    "suffix": None
                },
                "ultralytics": {
                    "import_success": False,
                    "error": None
                },
                "load": {
                    "success": False,
                    "task": None,
                    "classes": None,
                    "error": "Missing --model parameter",
                    "metadata": {}
                },
                "warnings": ["--model parameter is required"],
                "errors": ["Missing --model parameter"]
            }
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            parser.print_help()
            sys.exit(1)

    model_path = args.model
    resolved_path = os.path.abspath(model_path)
    exists = os.path.exists(resolved_path)
    size_bytes = os.path.getsize(resolved_path) if exists else None
    suffix = os.path.splitext(resolved_path)[1] if exists else os.path.splitext(model_path)[1]

    model_info = {
        "path": model_path,
        "resolved_path": resolved_path,
        "exists": exists,
        "size_bytes": size_bytes,
        "suffix": suffix
    }

    ultralytics_info = {
        "import_success": False,
        "error": None
    }

    load_info = {
        "success": False,
        "task": None,
        "classes": None,
        "error": None,
        "metadata": {}
    }

    # 1. Attempt to import ultralytics
    try:
        import ultralytics
        ultralytics_info["import_success"] = True
    except Exception as e:
        ultralytics_info["error"] = str(e)
        errors.append(f"Failed to import ultralytics: {e}")

    # 2. If exists and import succeeded, attempt to load the model
    if exists and ultralytics_info["import_success"]:
        # Set logging level for ultralytics to ERROR to suppress normal log output
        try:
            logging.getLogger("ultralytics").setLevel(logging.ERROR)
        except Exception:
            pass

        # Capture and suppress stdout/stderr output from the framework load process
        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
                from ultralytics import YOLO
                model = YOLO(resolved_path)
                
            load_info["success"] = True
            
            # Extract task and classes if available
            try:
                load_info["task"] = getattr(model, "task", None)
            except Exception as te:
                warnings.append(f"Failed to read model task: {te}")
                
            try:
                names = getattr(model, "names", None)
                if isinstance(names, dict):
                    # convert integer keys to strings for clean JSON representation
                    load_info["classes"] = {str(k): v for k, v in names.items()}
                else:
                    load_info["classes"] = names
            except Exception as ne:
                warnings.append(f"Failed to read model names: {ne}")
                
            # Cheap and stable metadata
            try:
                load_info["metadata"] = {
                    "type": type(model).__name__,
                    "overrides": getattr(model, "overrides", {})
                }
            except Exception as me:
                warnings.append(f"Failed to read model overrides/type: {me}")
                
        except Exception as le:
            load_info["error"] = str(le)
            errors.append(f"Failed to load model with YOLO: {le}")
            
            captured_stdout = f_stdout.getvalue().strip()
            captured_stderr = f_stderr.getvalue().strip()
            if captured_stdout:
                warnings.append(f"Captured stdout during load: {captured_stdout}")
            if captured_stderr:
                warnings.append(f"Captured stderr during load: {captured_stderr}")
    else:
        if not exists:
            warnings.append(f"Model file does not exist: {model_path}")
            load_info["error"] = f"File not found: {resolved_path}"
        elif not ultralytics_info["import_success"]:
            load_info["error"] = "Ultralytics package is not available"

    payload = {
        "model": model_info,
        "ultralytics": ultralytics_info,
        "load": load_info,
        "warnings": warnings,
        "errors": errors
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab Model Probe Summary")
        print("=" * 70)
        print(f"Model Path:      {model_info['path']}")
        print(f"Resolved Path:   {model_info['resolved_path']}")
        print(f"Exists:          {model_info['exists']}")
        if exists:
            print(f"File Size:       {model_info['size_bytes']} bytes")
            print(f"Suffix:          {model_info['suffix']}")
        print(f"Ultralytics:     {'Import Succeeded' if ultralytics_info['import_success'] else 'Import Failed'}")
        if ultralytics_info["error"]:
            print(f"  Error:         {ultralytics_info['error']}")
            
        print(f"Model Load:      {'Succeeded' if load_info['success'] else 'Failed'}")
        if load_info["error"]:
            print(f"  Error:         {load_info['error']}")
        if load_info["success"]:
            print(f"  Task:          {load_info['task']}")
            num_classes = len(load_info["classes"]) if isinstance(load_info["classes"], dict) else 0
            print(f"  Classes count: {num_classes}")
            if num_classes > 0:
                print(f"  Classes (first 5): {list(load_info['classes'].items())[:5]}...")
                
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
