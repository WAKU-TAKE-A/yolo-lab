import sys
import os
import json
import argparse
import contextlib
import io
import logging

def main():
    parser = argparse.ArgumentParser(description="AI-first Image Prediction Probe for YOLO-Lab")
    parser.add_argument("--model", type=str, help="Path to the model file")
    parser.add_argument("--image", type=str, help="Path to the input image file")
    parser.add_argument("--out", type=str, help="Output directory to save annotated image")
    parser.add_argument("--conf", type=float, help="Confidence threshold (optional)")
    parser.add_argument("--imgsz", type=int, help="Inference image size (optional)")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    args, unknown = parser.parse_known_args()

    warnings = []
    errors = []

    # Handle missing arguments in JSON mode cleanly
    if not args.model or not args.image or not args.out:
        missing_args = []
        if not args.model: missing_args.append("--model")
        if not args.image: missing_args.append("--image")
        if not args.out: missing_args.append("--out")
        
        err_msg = f"Missing required arguments: {', '.join(missing_args)}"
        if args.json:
            payload = {
                "model_path": args.model,
                "image_path": args.image,
                "out_dir": args.out,
                "annotated_image_path": None,
                "task": None,
                "image_size": None,
                "settings": {},
                "detections_count": 0,
                "detections": [],
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
    out_dir = args.out

    resolved_model_path = os.path.abspath(model_path)
    resolved_image_path = os.path.abspath(image_path)
    resolved_out_dir = os.path.abspath(out_dir)

    model_exists = os.path.exists(resolved_model_path)
    image_exists = os.path.exists(resolved_image_path)

    if not model_exists:
        errors.append(f"Model file does not exist: {model_path}")
    if not image_exists:
        errors.append(f"Image file does not exist: {image_path}")

    payload = {
        "model_path": model_path,
        "image_path": image_path,
        "out_dir": out_dir,
        "annotated_image_path": None,
        "task": None,
        "image_size": None,
        "settings": {
            "conf": args.conf,
            "imgsz": args.imgsz
        },
        "detections_count": 0,
        "detections": [],
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

            # Build predict parameters
            predict_kwargs = {"source": resolved_image_path, "save": False, "verbose": False}
            if args.conf is not None:
                predict_kwargs["conf"] = args.conf
            if args.imgsz is not None:
                predict_kwargs["imgsz"] = args.imgsz

            # Run prediction
            results = model.predict(**predict_kwargs)
            
        result = results[0]
        
        # orig_shape is (height, width)
        h, w = result.orig_shape
        payload["image_size"] = {"width": w, "height": h}

        # Detections extraction
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

        payload["detections"] = detections
        payload["detections_count"] = len(detections)

        # Mask summary
        if hasattr(result, "masks") and result.masks is not None:
            mask_shape = None
            if hasattr(result.masks, "data") and result.masks.data is not None:
                mask_shape = list(result.masks.data.shape)
            payload["masks_summary"] = {
                "count": len(result.masks),
                "shape": mask_shape
            }

        # Save annotated image
        os.makedirs(resolved_out_dir, exist_ok=True)
        annotated_img_name = f"annotated_{os.path.basename(image_path)}"
        annotated_path = os.path.join(resolved_out_dir, annotated_img_name)
        
        with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
            im_array = result.plot()
            import cv2
            cv2.imwrite(annotated_path, im_array)

        payload["annotated_image_path"] = os.path.abspath(annotated_path)

    except Exception as e:
        err_msg = f"Failed during prediction: {e}"
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
        print(" YOLO-Lab Image Predict Probe Summary")
        print("=" * 70)
        print(f"Model Path:      {payload['model_path']}")
        print(f"Image Path:      {payload['image_path']}")
        print(f"Annotated Path:  {payload['annotated_image_path']}")
        print(f"Task:            {payload['task']}")
        if payload['image_size']:
            print(f"Image Size:      {payload['image_size']['width']}x{payload['image_size']['height']}")
        print(f"Detections:      {payload['detections_count']}")
        
        if len(payload["detections"]) > 0:
            print("\nDetections (first 5):")
            for d in payload["detections"][:5]:
                mask_str = " (with mask)" if d["has_mask"] else ""
                print(f"  - {d['class_name']} ({d['confidence']:.2f}): {d['bbox_xyxy']}{mask_str}")
        
        if "masks_summary" in payload and payload["masks_summary"]:
            print(f"\nSegmentation Masks: {payload['masks_summary']['count']} (shape: {payload['masks_summary']['shape']})")

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
