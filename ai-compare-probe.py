import sys
import os
import json
import argparse
import contextlib
import io
import logging
import math

def safe_float(val):
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return "Infinity" if val > 0 else "-Infinity"
    return float(val)

def infer_onnx_kind(outputs_meta):
    if len(outputs_meta) == 1:
        shape = outputs_meta[0]["shape"]
        if len(shape) == 3:
            if shape[0] == 1 and shape[1] == 300 and shape[2] == 6:
                return "end2end_detections", f"Output shape {shape} matches the standard YOLO end-to-end NMS-free detection shape [batch, num_detections, box+conf+cls_id]."
            if shape[0] == 1 and shape[1] > 4 and shape[2] > 1000:
                return "raw_detection_head", f"Output shape {shape} matches the standard YOLO raw detection head shape [batch, box+cls_scores, num_anchors]."
    elif len(outputs_meta) >= 2:
        shapes = [o["shape"] for o in outputs_meta]
        has_3d = any(len(s) == 3 for s in shapes)
        has_4d = any(len(s) == 4 for s in shapes)
        if has_3d and has_4d:
            return "segmentation_head", f"Multiple outputs containing {shapes} indicate a raw instance segmentation head (e.g., bounding box/class coefficients plus prototype mask tensors)."
            
    return "unknown", f"Outputs { [o['shape'] for o in outputs_meta] } did not match recognized YOLO templates."

def main():
    parser = argparse.ArgumentParser(description="AI-first Comparison Probe for YOLO-Lab")
    parser.add_argument("--pt", type=str, help="Path to the PyTorch model file (.pt)")
    parser.add_argument("--onnx", type=str, help="Path to the ONNX model file (.onnx)")
    parser.add_argument("--image", type=str, help="Path to the input image file")
    parser.add_argument("--out", type=str, help="Output directory to save PyTorch annotated image")
    parser.add_argument("--conf", type=float, help="Confidence threshold (optional)")
    parser.add_argument("--imgsz", type=int, help="Inference image size (optional)")
    parser.add_argument("--providers", type=str, help="Comma-separated list of execution providers (optional)")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    args, unknown = parser.parse_known_args()

    warnings = []
    errors = []

    # Handle missing arguments in JSON mode cleanly
    if not args.pt or not args.onnx or not args.image:
        missing_args = []
        if not args.pt: missing_args.append("--pt")
        if not args.onnx: missing_args.append("--onnx")
        if not args.image: missing_args.append("--image")
        
        err_msg = f"Missing required arguments: {', '.join(missing_args)}"
        if args.json:
            payload = {
                "pt_model_path": args.pt,
                "onnx_model_path": args.onnx,
                "image_path": args.image,
                "out_dir": args.out,
                "relationship": {},
                "pt_prediction": {},
                "onnx_raw_inference": {},
                "warnings": [err_msg],
                "errors": [err_msg]
            }
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(err_msg, file=sys.stderr)
            parser.print_help()
            sys.exit(1)

    pt_path = args.pt
    onnx_path = args.onnx
    image_path = args.image
    out_dir = args.out

    resolved_pt = os.path.abspath(pt_path)
    resolved_onnx = os.path.abspath(onnx_path)
    resolved_image = os.path.abspath(image_path)

    if not os.path.exists(resolved_pt):
        errors.append(f"PyTorch model does not exist: {pt_path}")
    if not os.path.exists(resolved_onnx):
        errors.append(f"ONNX model does not exist: {onnx_path}")
    if not os.path.exists(resolved_image):
        errors.append(f"Image file does not exist: {image_path}")

    payload = {
        "pt_model_path": pt_path,
        "onnx_model_path": onnx_path,
        "image_path": image_path,
        "out_dir": out_dir,
        "relationship": {},
        "pt_prediction": {
            "task": None,
            "detections_count": 0,
            "detections": [],
            "masks_summary": None,
            "annotated_image_path": None
        },
        "onnx_raw_inference": {
            "session_success": False,
            "session_providers": [],
            "input_metadata": None,
            "prepared_input": None,
            "outputs": []
        },
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

    # Import dependencies
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
        from ultralytics import YOLO
    except Exception as e:
        err_msg = f"Failed to import required libraries: {e}"
        errors.append(err_msg)
        payload["errors"] = errors
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(err_msg, file=sys.stderr)
            sys.exit(1)

    # Suppress Ultralytics and ORT logs
    try:
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
    except Exception:
        pass

    f_stdout = io.StringIO()
    f_stderr = io.StringIO()

    # 1. Run PyTorch inference
    try:
        with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
            pt_model = YOLO(resolved_pt)
            payload["pt_prediction"]["task"] = getattr(pt_model, "task", None)

            # Predict config
            predict_kwargs = {"source": resolved_image, "save": False, "verbose": False}
            if args.conf is not None:
                predict_kwargs["conf"] = args.conf
            if args.imgsz is not None:
                predict_kwargs["imgsz"] = args.imgsz

            results = pt_model.predict(**predict_kwargs)
            
        result = results[0]

        # Extract PyTorch detections
        detections = []
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names[cls_id] if result.names and cls_id in result.names else f"class_{cls_id}"
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().tolist()
                
                has_mask = False
                if hasattr(result, "masks") and result.masks is not None:
                    has_mask = len(result.masks) > i
                    
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf_val,
                    "bbox_xyxy": xyxy,
                    "has_mask": has_mask
                })

        payload["pt_prediction"]["detections"] = detections
        payload["pt_prediction"]["detections_count"] = len(detections)

        # PyTorch mask summary
        if hasattr(result, "masks") and result.masks is not None:
            mask_shape = None
            if hasattr(result.masks, "data") and result.masks.data is not None:
                mask_shape = list(result.masks.data.shape)
            payload["pt_prediction"]["masks_summary"] = {
                "count": len(result.masks),
                "shape": mask_shape
            }

        # Save PyTorch annotated image if requested
        if out_dir:
            resolved_out = os.path.abspath(out_dir)
            os.makedirs(resolved_out, exist_ok=True)
            annotated_img_name = f"annotated_pt_{os.path.basename(image_path)}"
            annotated_path = os.path.join(resolved_out, annotated_img_name)
            
            with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
                im_array = result.plot()
                cv2.imwrite(annotated_path, im_array)

            payload["pt_prediction"]["annotated_image_path"] = os.path.abspath(annotated_path)

    except Exception as e:
        warnings.append(f"PyTorch prediction failed: {e}")

    # 2. Run ONNX raw inference
    try:
        # Load image for ORT
        img = cv2.imread(resolved_image)
        if img is None:
            raise ValueError(f"OpenCV failed to read image: {image_path}")
        
        orig_h, orig_w = img.shape[:2]

        # ORT session options
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        
        providers_list = None
        if args.providers:
            providers_list = [p.strip() for p in args.providers.split(",")]

        if providers_list:
            session = ort.InferenceSession(resolved_onnx, sess_options=opts, providers=providers_list)
        else:
            session = ort.InferenceSession(resolved_onnx, sess_options=opts)

        payload["onnx_raw_inference"]["session_success"] = True
        payload["onnx_raw_inference"]["session_providers"] = session.get_providers()

        # Parse first input details
        inputs = session.get_inputs()
        first_input = inputs[0]
        input_name = first_input.name
        input_shape = first_input.shape
        input_type = first_input.type

        payload["onnx_raw_inference"]["input_metadata"] = {
            "name": input_name,
            "shape": input_shape,
            "type": input_type
        }

        # Resolve imgsz
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

        # Preprocess standard YOLO NCHW
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (in_w, in_h))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_nchw = np.expand_dims(img_chw, axis=0)

        payload["onnx_raw_inference"]["prepared_input"] = {
            "shape": list(img_nchw.shape),
            "dtype": str(img_nchw.dtype),
            "min": safe_float(np.min(img_nchw)),
            "max": safe_float(np.max(img_nchw)),
            "mean": safe_float(np.mean(img_nchw))
        }

        # Run session
        output_names = [out.name for out in session.get_outputs()]
        raw_outputs = session.run(output_names, {input_name: img_nchw})

        # Process outputs metadata & stats
        outputs_list = []
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

        payload["onnx_raw_inference"]["outputs"] = outputs_list

        # Infer relationship and output kind
        onnx_outputs_meta = [{"name": o["name"], "shape": o["shape"], "type": o["dtype"]} for o in outputs_list]
        onnx_kind, explanation = infer_onnx_kind(onnx_outputs_meta)

        payload["relationship"] = {
            "pt_task": payload["pt_prediction"]["task"],
            "onnx_output_shapes": {o["name"]: o["shape"] for o in outputs_list},
            "coarse_onnx_kind": onnx_kind,
            "explanation": explanation
        }

    except Exception as e:
        warnings.append(f"ONNX raw inference failed: {e}")

    # Log captured messages to warnings if any
    captured_stdout = f_stdout.getvalue().strip()
    captured_stderr = f_stderr.getvalue().strip()
    if captured_stdout:
        warnings.append(f"Captured stdout during execution: {captured_stdout}")
    if captured_stderr:
        warnings.append(f"Captured stderr during execution: {captured_stderr}")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab PyTorch vs ONNX Comparison Summary")
        print("=" * 70)
        print(f"PT Model Path:   {payload['pt_model_path']}")
        print(f"ONNX Model Path: {payload['onnx_model_path']}")
        print(f"Image Path:      {payload['image_path']}")
        print(f"Relationship:")
        if payload['relationship']:
            print(f"  PT Task:       {payload['relationship']['pt_task']}")
            print(f"  ONNX Output:   {payload['relationship']['onnx_output_shapes']}")
            print(f"  Coarse Kind:   {payload['relationship']['coarse_onnx_kind']}")
            print(f"  Explanation:   {payload['relationship']['explanation']}")
        
        print("\nPyTorch Prediction:")
        print(f"  Task:          {payload['pt_prediction']['task']}")
        print(f"  Detections:    {payload['pt_prediction']['detections_count']}")
        if len(payload['pt_prediction']['detections']) > 0:
            for d in payload['pt_prediction']['detections'][:3]:
                print(f"    - {d['class_name']} ({d['confidence']:.2f}): {d['bbox_xyxy']}")
        if payload['pt_prediction']['masks_summary']:
            print(f"  Masks:         {payload['pt_prediction']['masks_summary']}")

        print("\nONNX Raw Inference:")
        print(f"  Load Success:  {payload['onnx_raw_inference']['session_success']}")
        print(f"  Providers:     {', '.join(payload['onnx_raw_inference']['session_providers'])}")
        if payload['onnx_raw_inference']['session_success']:
            for o in payload['onnx_raw_inference']['outputs']:
                print(f"    - {o['name']}: shape={o['shape']}, min={o['min']:.4f}, max={o['max']:.4f}")
                print(f"      sample values: {o['sample_values']}")

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
