import sys
import os
import json
import argparse
import math

def safe_float(val):
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return "Infinity" if val > 0 else "-Infinity"
    return float(val)

def main():
    parser = argparse.ArgumentParser(description="AI-first ONNX Raw Inference Probe for YOLO-Lab")
    parser.add_argument("--model", type=str, help="Path to the ONNX model file")
    parser.add_argument("--image", type=str, help="Path to the input image file")
    parser.add_argument("--providers", type=str, help="Comma-separated list of execution providers to try")
    parser.add_argument("--imgsz", type=int, help="Optional inference size (override default 640)")
    parser.add_argument("--save-raw", action="store_true", help="Save raw outputs as .npy files")
    parser.add_argument("--out", type=str, help="Output directory to save raw outputs")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    args, unknown = parser.parse_known_args()

    warnings = []
    errors = []

    # Handle missing arguments in JSON mode cleanly
    if not args.model or not args.image:
        missing_args = []
        if not args.model: missing_args.append("--model")
        if not args.image: missing_args.append("--image")
        
        err_msg = f"Missing required arguments: {', '.join(missing_args)}"
        if args.json:
            payload = {
                "model_path": args.model,
                "image_path": args.image,
                "image_orig_size": None,
                "image_preprocessed_size": None,
                "onnxruntime": {
                    "import_success": False,
                    "version": None,
                    "available_providers": [],
                    "requested_providers": None,
                    "session_success": False,
                    "session_providers": []
                },
                "input_metadata": None,
                "prepared_input": None,
                "outputs": [],
                "saved_raw_output_paths": {},
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
    image_path = args.image

    resolved_model_path = os.path.abspath(model_path)
    resolved_image_path = os.path.abspath(image_path)

    model_exists = os.path.exists(resolved_model_path)
    image_exists = os.path.exists(resolved_image_path)

    if not model_exists:
        errors.append(f"Model file does not exist: {model_path}")
    if not image_exists:
        errors.append(f"Image file does not exist: {image_path}")

    payload = {
        "model_path": model_path,
        "image_path": image_path,
        "image_orig_size": None,
        "image_preprocessed_size": None,
        "onnxruntime": {
            "import_success": False,
            "version": None,
            "available_providers": [],
            "requested_providers": None,
            "session_success": False,
            "session_providers": []
        },
        "input_metadata": None,
        "prepared_input": None,
        "outputs": [],
        "saved_raw_output_paths": {},
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

    # 1. Try imports
    try:
        import onnxruntime as ort
        import cv2
        import numpy as np
        payload["onnxruntime"]["import_success"] = True
        payload["onnxruntime"]["version"] = ort.__version__
        payload["onnxruntime"]["available_providers"] = ort.get_available_providers()
    except Exception as e:
        err_msg = f"Failed to import required libraries (onnxruntime, opencv, numpy): {e}"
        errors.append(err_msg)
        payload["errors"] = errors
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(err_msg, file=sys.stderr)
            sys.exit(1)

    # Setup requested providers
    providers_list = None
    if args.providers:
        providers_list = [p.strip() for p in args.providers.split(",")]
        payload["onnxruntime"]["requested_providers"] = providers_list

    try:
        # Load image to inspect size
        img = cv2.imread(resolved_image_path)
        if img is None:
            raise ValueError(f"OpenCV failed to read image: {image_path}")
        
        orig_h, orig_w = img.shape[:2]
        payload["image_orig_size"] = {"width": orig_w, "height": orig_h}

        # Create session
        opts = ort.SessionOptions()
        opts.log_severity_level = 3 # Suppress verbose logs
        
        if providers_list:
            session = ort.InferenceSession(resolved_model_path, sess_options=opts, providers=providers_list)
        else:
            session = ort.InferenceSession(resolved_model_path, sess_options=opts)

        payload["onnxruntime"]["session_success"] = True
        payload["onnxruntime"]["session_providers"] = session.get_providers()

        # Parse first input details
        inputs = session.get_inputs()
        first_input = inputs[0]
        input_name = first_input.name
        input_shape = first_input.shape
        input_type = first_input.type

        payload["input_metadata"] = {
            "name": input_name,
            "shape": input_shape,
            "type": input_type
        }

        # Determine target imgsz
        in_h = 640
        in_w = 640
        if len(input_shape) == 4:
            if isinstance(input_shape[2], int):
                in_h = input_shape[2]
            if isinstance(input_shape[3], int):
                in_w = input_shape[3]

        if args.imgsz is not None:
            in_h = args.imgsz
            in_w = args.imgsz

        payload["image_preprocessed_size"] = {"width": in_w, "height": in_h}

        # Preprocess image standard YOLO-style
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (in_w, in_h))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_nchw = np.expand_dims(img_chw, axis=0)

        # Record prepared input statistics
        payload["prepared_input"] = {
            "shape": list(img_nchw.shape),
            "dtype": str(img_nchw.dtype),
            "min": safe_float(np.min(img_nchw)),
            "max": safe_float(np.max(img_nchw)),
            "mean": safe_float(np.mean(img_nchw))
        }

        # Run session
        output_names = [out.name for out in session.get_outputs()]
        raw_outputs = session.run(output_names, {input_name: img_nchw})

        # Process outputs
        outputs_list = []
        saved_paths = {}

        # Handle save directories if needed
        save_dir = None
        if args.save_raw:
            save_dir = args.out if args.out else "runs/onnx_raw_probe"
            os.makedirs(save_dir, exist_ok=True)

        for name, arr in zip(output_names, raw_outputs):
            flat_arr = arr.flatten()
            sample_vals = [safe_float(x) for x in flat_arr[:5].tolist()]
            
            out_meta = {
                "name": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": safe_float(np.min(arr)),
                "max": safe_float(np.max(arr)),
                "mean": safe_float(np.mean(arr)),
                "sample_values": sample_vals
            }
            outputs_list.append(out_meta)

            if args.save_raw and save_dir:
                clean_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in name])
                filename = f"{clean_name}.npy"
                file_path = os.path.join(save_dir, filename)
                np.save(file_path, arr)
                saved_paths[name] = os.path.abspath(file_path)

        payload["outputs"] = outputs_list
        payload["saved_raw_output_paths"] = saved_paths

    except Exception as e:
        err_msg = f"Failed during ONNX raw inference execution: {e}"
        errors.append(err_msg)
        payload["errors"] = errors

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab ONNX Raw Inference Probe Summary")
        print("=" * 70)
        print(f"Model Path:      {payload['model_path']}")
        print(f"Image Path:      {payload['image_path']}")
        print(f"Session Load:    {'Succeeded' if payload['onnxruntime']['session_success'] else 'Failed'}")
        print(f"Actual Prov:     {', '.join(payload['onnxruntime']['session_providers'])}")
        
        if payload['image_orig_size']:
            print(f"Image Size:      {payload['image_orig_size']['width']}x{payload['image_orig_size']['height']} -> Preprocessed: {payload['image_preprocessed_size']['width']}x{payload['image_preprocessed_size']['height']}")

        if payload['prepared_input']:
            print(f"Prepared Input:  shape={payload['prepared_input']['shape']}, dtype={payload['prepared_input']['dtype']}, min={payload['prepared_input']['min']:.4f}, max={payload['prepared_input']['max']:.4f}")

        if len(payload["outputs"]) > 0:
            print("\nRaw Outputs Statistics:")
            for o in payload["outputs"]:
                print(f"  - {o['name']}: shape={o['shape']}, dtype={o['dtype']}")
                print(f"    min={o['min']:.4f}, max={o['max']:.4f}, mean={o['mean']:.4f}")
                print(f"    sample values (first 5 flattened): {o['sample_values']}")

        if payload["saved_raw_output_paths"]:
            print("\nSaved raw (.npy) output paths:")
            for k, v in payload["saved_raw_output_paths"].items():
                print(f"  {k}: {v}")

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
