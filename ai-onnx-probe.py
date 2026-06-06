import sys
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="AI-first ONNX Model Probe for YOLO-Lab")
    parser.add_argument("--model", type=str, help="Path to the ONNX model file to inspect")
    parser.add_argument("--providers", type=str, help="Comma-separated list of execution providers to try")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    args, unknown = parser.parse_known_args()

    warnings = []
    errors = []

    # Handle missing arguments in JSON mode cleanly
    if not args.model:
        err_msg = "Missing required argument: --model"
        if args.json:
            payload = {
                "model": {
                    "path": None,
                    "resolved_path": None,
                    "exists": False,
                    "size_bytes": None
                },
                "onnx": {
                    "import_success": False,
                    "load_success": False,
                    "check_status": None,
                    "check_error": None,
                    "opsets": [],
                    "error": None
                },
                "onnxruntime": {
                    "import_success": False,
                    "version": None,
                    "available_providers": [],
                    "requested_providers": None,
                    "session_success": False,
                    "session_providers": [],
                    "session_error": None,
                    "inputs": [],
                    "outputs": []
                },
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
    resolved_path = os.path.abspath(model_path)
    exists = os.path.exists(resolved_path)
    size_bytes = os.path.getsize(resolved_path) if exists else None

    model_info = {
        "path": model_path,
        "resolved_path": resolved_path,
        "exists": exists,
        "size_bytes": size_bytes
    }

    onnx_info = {
        "import_success": False,
        "load_success": False,
        "check_status": None,
        "check_error": None,
        "opsets": [],
        "error": None
    }

    onnxruntime_info = {
        "import_success": False,
        "version": None,
        "available_providers": [],
        "requested_providers": None,
        "session_success": False,
        "session_providers": [],
        "session_error": None,
        "inputs": [],
        "outputs": []
    }

    if not exists:
        err_msg = f"Model file does not exist: {model_path}"
        errors.append(err_msg)
        onnx_info["error"] = f"File not found: {resolved_path}"
        onnxruntime_info["session_error"] = f"File not found: {resolved_path}"
    
    # 1. Inspect with onnx library
    if exists:
        try:
            import onnx
            onnx_info["import_success"] = True
            try:
                onnx_model = onnx.load(resolved_path)
                onnx_info["load_success"] = True
                
                # Check model
                try:
                    onnx.checker.check_model(onnx_model)
                    onnx_info["check_status"] = True
                except Exception as ce:
                    onnx_info["check_status"] = False
                    onnx_info["check_error"] = str(ce)
                    warnings.append(f"onnx.checker warning: {ce}")
                    
                # Extract opset imports
                for opset in onnx_model.opset_import:
                    onnx_info["opsets"].append({
                        "domain": opset.domain,
                        "version": opset.version
                    })
            except Exception as le:
                onnx_info["error"] = str(le)
                errors.append(f"Failed to load ONNX model structure: {le}")
        except Exception as e:
            warnings.append(f"onnx library not available: {e}")

    # 2. Inspect with onnxruntime library
    if exists:
        try:
            import onnxruntime as ort
            onnxruntime_info["import_success"] = True
            onnxruntime_info["version"] = ort.__version__
            onnxruntime_info["available_providers"] = ort.get_available_providers()
            
            # Setup requested providers
            providers_list = None
            if args.providers:
                providers_list = [p.strip() for p in args.providers.split(",")]
                onnxruntime_info["requested_providers"] = providers_list

            # Create InferenceSession
            try:
                # Suppress output logs
                opts = ort.SessionOptions()
                opts.log_severity_level = 3 # Warning level
                
                if providers_list:
                    session = ort.InferenceSession(resolved_path, sess_options=opts, providers=providers_list)
                else:
                    session = ort.InferenceSession(resolved_path, sess_options=opts)
                    
                onnxruntime_info["session_success"] = True
                onnxruntime_info["session_providers"] = session.get_providers()
                
                # Get inputs info
                for inp in session.get_inputs():
                    onnxruntime_info["inputs"].append({
                        "name": inp.name,
                        "shape": inp.shape,
                        "type": inp.type
                    })
                    
                # Get outputs info
                for out in session.get_outputs():
                    onnxruntime_info["outputs"].append({
                        "name": out.name,
                        "shape": out.shape,
                        "type": out.type
                    })
            except Exception as se:
                onnxruntime_info["session_error"] = str(se)
                errors.append(f"ONNXRuntime session creation failed: {se}")
        except Exception as e:
            err_msg = f"onnxruntime library not available: {e}"
            errors.append(err_msg)
            onnxruntime_info["session_error"] = err_msg

    payload = {
        "model": model_info,
        "onnx": onnx_info,
        "onnxruntime": onnxruntime_info,
        "warnings": warnings,
        "errors": errors
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab ONNX Probe Summary")
        print("=" * 70)
        print(f"Model Path:      {model_info['path']}")
        print(f"Resolved Path:   {model_info['resolved_path']}")
        print(f"Exists:          {model_info['exists']}")
        if exists:
            print(f"File Size:       {model_info['size_bytes']} bytes")
            
        print(f"ONNX Package:    {'Available' if onnx_info['import_success'] else 'Not Available'}")
        if onnx_info['load_success']:
            opset_strs = [f"{o['domain']} v{o['version']}" if o['domain'] else f"v{o['version']}" for o in onnx_info['opsets']]
            print(f"  Opsets:        {', '.join(opset_strs)}")
            print(f"  Check Status:  {'Success' if onnx_info['check_status'] else 'Failed'}")
            if onnx_info['check_error']:
                print(f"    Check Error: {onnx_info['check_error']}")
        elif onnx_info['error']:
            print(f"  Error:         {onnx_info['error']}")

        print(f"ONNXRuntime:     {'Available' if onnxruntime_info['import_success'] else 'Not Available'}")
        if onnxruntime_info['import_success']:
            print(f"  ORT Version:   {onnxruntime_info['version']}")
            print(f"  Session Load:  {'Succeeded' if onnxruntime_info['session_success'] else 'Failed'}")
            if onnxruntime_info['session_error']:
                print(f"    Load Error:  {onnxruntime_info['session_error']}")
            print(f"  Actual Prov:   {', '.join(onnxruntime_info['session_providers'])}")
            
            if onnxruntime_info['session_success']:
                print("\nInputs:")
                for inp in onnxruntime_info['inputs']:
                    print(f"  - {inp['name']}: shape={inp['shape']}, type={inp['type']}")
                print("\nOutputs:")
                for out in onnxruntime_info['outputs']:
                    print(f"  - {out['name']}: shape={out['shape']}, type={out['type']}")

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
