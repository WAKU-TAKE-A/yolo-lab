import sys
import os
import json
import argparse
import csv
import shutil
import logging
import contextlib
import io
from datetime import datetime, timezone, timedelta

def run_evaluate(args):
    warnings = []
    errors = []

    model_path = args.model
    input_dir = args.input
    out_dir = args.out

    resolved_model = os.path.abspath(model_path)
    resolved_input = os.path.abspath(input_dir)
    resolved_out = os.path.normpath(os.path.abspath(out_dir))

    # Validate paths
    if not os.path.exists(resolved_model):
        errors.append(f"Model file does not exist: {model_path}")
    if not os.path.exists(resolved_input):
        errors.append(f"Input directory does not exist: {input_dir}")
    elif not os.path.isdir(resolved_input):
        errors.append(f"Input path is not a directory: {input_dir}")

    eval_id = os.path.basename(resolved_out)

    payload = {
        "eval_id": eval_id,
        "out_dir": out_dir,
        "image_count": 0,
        "detection_count": 0,
        "manifest_path": None,
        "results_csv_path": None,
        "review_jsonl_path": None,
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

    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except Exception as e:
        err_msg = f"Failed to import required libraries: {e}"
        errors.append(err_msg)
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(err_msg, file=sys.stderr)
            sys.exit(1)

    # Suppress Ultralytics verbose log prints
    try:
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
    except Exception:
        pass

    # Create directories
    os.makedirs(resolved_out, exist_ok=True)
    images_out_dir = os.path.join(resolved_out, "images")
    overlays_out_dir = os.path.join(resolved_out, "overlays")
    predictions_out_dir = os.path.join(resolved_out, "predictions")

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(overlays_out_dir, exist_ok=True)
    os.makedirs(predictions_out_dir, exist_ok=True)

    # Scan image folder
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = []
    try:
        for f in os.listdir(resolved_input):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                image_files.append(f)
        image_files.sort()
    except Exception as e:
        err_msg = f"Failed to read input directory: {e}"
        errors.append(err_msg)
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(err_msg, file=sys.stderr)
            sys.exit(1)

    if not image_files:
        warnings.append(f"No valid image files found in input directory: {input_dir}")

    # Load YOLO model
    f_stdout = io.StringIO()
    f_stderr = io.StringIO()
    model = None
    try:
        with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
            model = YOLO(resolved_model)
    except Exception as e:
        err_msg = f"Failed to load YOLO model: {e}"
        errors.append(err_msg)
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(err_msg, file=sys.stderr)
            sys.exit(1)

    # Initialize results.csv
    csv_path = os.path.join(resolved_out, "results.csv")
    csv_file = None
    csv_writer = None
    try:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["eval_id", "det_id", "image_id", "source_image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
    except Exception as e:
        errors.append(f"Failed to create results.csv: {e}")
        if csv_file: csv_file.close()
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(f"Failed to create results.csv: {e}", file=sys.stderr)
            sys.exit(1)

    # Initialize empty review.jsonl
    review_jsonl_path = os.path.join(resolved_out, "review.jsonl")
    if not os.path.exists(review_jsonl_path):
        try:
            with open(review_jsonl_path, "w", encoding="utf-8") as rf:
                pass
        except Exception as e:
            warnings.append(f"Failed to create empty review.jsonl: {e}")

    global_det_id = 1
    processed_images = 0

    for idx, fname in enumerate(image_files):
        image_id = f"{idx + 1:06d}"
        src_img_path = os.path.join(resolved_input, fname)
        ext = os.path.splitext(fname)[1]
        dest_img_path = os.path.join(images_out_dir, f"{image_id}{ext}")
        dest_overlay_path = os.path.join(overlays_out_dir, f"{image_id}_result.jpg")
        dest_pred_path = os.path.join(predictions_out_dir, f"{image_id}.json")

        # Copy original image
        try:
            shutil.copy2(src_img_path, dest_img_path)
        except Exception as e:
            warnings.append(f"Failed to copy image {fname} to runs: {e}")
            continue

        # Inference
        try:
            predict_kwargs = {"source": src_img_path, "save": False, "verbose": False}
            if args.conf is not None:
                predict_kwargs["conf"] = args.conf
            if args.imgsz is not None:
                predict_kwargs["imgsz"] = args.imgsz

            with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
                results = model.predict(**predict_kwargs)
            result = results[0]
        except Exception as e:
            warnings.append(f"Prediction failed on image {fname}: {e}")
            continue

        # Extract instances
        img_detections = []
        
        # Check available geometry
        has_boxes = hasattr(result, "boxes") and result.boxes is not None
        has_masks = hasattr(result, "masks") and result.masks is not None
        has_obb = hasattr(result, "obb") and result.obb is not None
        requested_geometry = args.geometry
        save_polygon = requested_geometry in ("auto", "polygon")
        save_obb = requested_geometry in ("auto", "obb")
        
        # Use OBB if present, else fallback to boxes
        if has_obb:
            obbs = result.obb
            for i in range(len(obbs)):
                cls_id = int(obbs.cls[i].item())
                cls_name = result.names[cls_id] if result.names and cls_id in result.names else f"class_{cls_id}"
                conf_val = float(obbs.conf[i].item())
                xyxy = obbs.xyxy[i].cpu().tolist() # axis-aligned bounding box

                det = {
                    "det_id": global_det_id,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf_val,
                    "bbox_xyxy": xyxy
                }
                if save_obb:
                    det["geometry_type"] = "obb"
                    det["geometry_coordinates"] = "image_pixels"
                    det["obb_xywhr"] = obbs.xywhr[i].cpu().tolist()
                    det["obb_xyxyxyxy"] = obbs.xyxyxyxy[i].cpu().tolist() # 4 corners
                img_detections.append(det)

                # Write to CSV
                x1, y1, x2, y2 = xyxy
                csv_writer.writerow([
                    eval_id, global_det_id, image_id, f"{input_dir}/{fname}",
                    cls_id, cls_name, f"{conf_val:.4f}",
                    f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"
                ])

                global_det_id += 1
        elif has_boxes:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names[cls_id] if result.names and cls_id in result.names else f"class_{cls_id}"
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().tolist()

                det = {
                    "det_id": global_det_id,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf_val,
                    "bbox_xyxy": xyxy
                }
                
                # Add segmentation polygon if requested and available.
                if save_polygon and has_masks and i < len(result.masks.xy):
                    det["geometry_type"] = "polygon"
                    det["geometry_coordinates"] = "image_pixels"
                    det["polygon_xy"] = result.masks.xy[i].tolist()
                    if hasattr(result.masks, "xyn") and i < len(result.masks.xyn):
                        det["polygon_xyn"] = result.masks.xyn[i].tolist()

                img_detections.append(det)

                # Write to CSV
                x1, y1, x2, y2 = xyxy
                csv_writer.writerow([
                    eval_id, global_det_id, image_id, f"{input_dir}/{fname}",
                    cls_id, cls_name, f"{conf_val:.4f}",
                    f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"
                ])

                global_det_id += 1

        # Draw overlay image
        try:
            img_data = cv2.imread(src_img_path)
            if img_data is not None:
                # Top level evaluation text
                text_run = f"eval: {eval_id} image: {image_id}"
                cv2.putText(img_data, text_run, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img_data, text_run, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                for det in img_detections:
                    d_id = det["det_id"]
                    cls_name = det["class_name"]
                    conf = det["confidence"]
                    x1, y1, x2, y2 = [int(round(coord)) for coord in det["bbox_xyxy"]]

                    # Draw geometry
                    color = (0, 255, 0)
                    draw_bbox = False
                    draw_poly = False
                    draw_obb = False
                    
                    req_overlay = args.overlay
                    if req_overlay == "auto":
                        if "obb_xyxyxyxy" in det: draw_obb = True
                        elif "polygon_xy" in det: draw_poly = True
                        else: draw_bbox = True
                    elif req_overlay == "bbox": draw_bbox = True
                    elif req_overlay == "polygon" or req_overlay == "mask": draw_poly = True
                    elif req_overlay == "obb": draw_obb = True
                    elif req_overlay == "both":
                        draw_bbox = True
                        if "obb_xyxyxyxy" in det: draw_obb = True
                        if "polygon_xy" in det: draw_poly = True

                    if draw_poly and "polygon_xy" in det:
                        pts = np.array(det["polygon_xy"], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        if req_overlay == "mask":
                            overlay_img = img_data.copy()
                            cv2.fillPoly(overlay_img, [pts], color)
                            cv2.addWeighted(overlay_img, 0.4, img_data, 0.6, 0, img_data)
                        else:
                            cv2.polylines(img_data, [pts], True, color, 2)
                    
                    if draw_obb and "obb_xyxyxyxy" in det:
                        pts = np.array(det["obb_xyxyxyxy"], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(img_data, [pts], True, color, 2)
                        
                    if draw_bbox or (not draw_poly and not draw_obb):
                        cv2.rectangle(img_data, (x1, y1), (x2, y2), color, 2)

                    # Draw label: [det_id] class_name conf
                    label = f"[{d_id}] {cls_name} {conf:.2f}"
                    (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    y_text = y1 - 4
                    if y_text - lbl_h < 0:
                        cv2.rectangle(img_data, (x1, y1), (x1 + lbl_w + 4, y1 + lbl_h + 4), color, -1)
                        cv2.putText(img_data, label, (x1 + 2, y1 + lbl_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(img_data, (x1, y1 - lbl_h - 4), (x1 + lbl_w + 4, y1), color, -1)
                        cv2.putText(img_data, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.imwrite(dest_overlay_path, img_data)
            else:
                warnings.append(f"Failed to read image for overlay: {src_img_path}")
        except Exception as e:
            warnings.append(f"Failed to draw overlay for image {fname}: {e}")

        # Save prediction json
        try:
            pred_data = {
                "eval_id": eval_id,
                "image_id": image_id,
                "source_image": f"{input_dir}/{fname}",
                "overlay_image": f"overlays/{image_id}_result.jpg",
                "detections": img_detections
            }
            with open(dest_pred_path, "w", encoding="utf-8") as pf:
                json.dump(pred_data, pf, indent=2)
        except Exception as e:
            warnings.append(f"Failed to save prediction JSON for image {fname}: {e}")

        processed_images += 1

    # Close CSV
    csv_file.close()

    # Write manifest.json
    manifest_path = os.path.join(resolved_out, "manifest.json")
    try:
        tz = timezone(timedelta(hours=9)) # JST
        manifest_data = {
            "eval_id": eval_id,
            "created_at": datetime.now(tz).isoformat(),
            "model": model_path,
            "input": input_dir,
            "task": getattr(model, "task", None) if model else "unknown",
            "conf": args.conf if args.conf is not None else 0.25,
            "imgsz": args.imgsz if args.imgsz is not None else 640,
            "image_count": processed_images,
            "detection_count": global_det_id - 1
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, indent=2)
    except Exception as e:
        errors.append(f"Failed to write manifest.json: {e}")

    # Final payload values
    payload["image_count"] = processed_images
    payload["detection_count"] = global_det_id - 1
    payload["manifest_path"] = os.path.relpath(manifest_path, os.getcwd()) if os.path.exists(manifest_path) else None
    payload["results_csv_path"] = os.path.relpath(csv_path, os.getcwd()) if os.path.exists(csv_path) else None
    payload["review_jsonl_path"] = os.path.relpath(review_jsonl_path, os.getcwd()) if os.path.exists(review_jsonl_path) else None

    # Log captured messages to warnings
    captured_stdout = f_stdout.getvalue().strip()
    captured_stderr = f_stderr.getvalue().strip()
    if captured_stdout:
        warnings.append(f"Captured stdout during evaluation: {captured_stdout}")
    if captured_stderr:
        warnings.append(f"Captured stderr during evaluation: {captured_stderr}")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab Evaluation Summary")
        print("=" * 70)
        print(f"Eval ID:         {payload['eval_id']}")
        print(f"Output Run Dir:  {payload['out_dir']}")
        print(f"Images count:    {payload['image_count']}")
        print(f"Detections cnt:  {payload['detection_count']}")
        print(f"Manifest Path:   {payload['manifest_path']}")
        print(f"Results CSV:     {payload['results_csv_path']}")
        print(f"Review JSONL:    {payload['review_jsonl_path']}")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  - {e}")
        print("=" * 70)

def run_show(args):
    warnings = []
    errors = []

    eval_id = args.eval_id
    det_id = None
    try:
        det_id = int(args.det_id)
    except ValueError:
        errors.append(f"Invalid det_id (must be an integer): {args.det_id}")

    runs_root = args.runs_root or "runs"
    run_dir = os.path.normpath(os.path.join(runs_root, eval_id))

    payload = {
        "eval_id": eval_id,
        "det_id": det_id,
        "run_dir": run_dir,
        "manifest": None,
        "result_row": None,
        "source_image_path": None,
        "copied_image_path": None,
        "overlay_image_path": None,
        "prediction_json_path": None,
        "prediction_detection": None,
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

    # 1. Check if run directory exists
    if not os.path.exists(run_dir):
        errors.append(f"Run directory does not exist: {run_dir}")
    elif not os.path.isdir(run_dir):
        errors.append(f"Run path is not a directory: {run_dir}")

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    # 2. Read manifest.json (if present)
    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload["manifest"] = json.load(f)
        except Exception as e:
            warnings.append(f"Failed to read manifest.json: {e}")
    else:
        warnings.append(f"manifest.json not found in run directory")

    # 3. Read results.csv
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        errors.append(f"results.csv not found in run directory: {run_dir}")
    else:
        # Find det_id row
        try:
            matching_row = None
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                actual_cols = reader.fieldnames if reader.fieldnames else []
                # Check that det_id exists in columns
                if "det_id" not in actual_cols:
                    errors.append(f"results.csv is missing 'det_id' column")
                else:
                    for row in reader:
                        if not row:
                            continue
                        try:
                            row_det = row.get("det_id")
                            if row_det is not None and int(row_det) == det_id:
                                matching_row = row
                                break
                        except (ValueError, TypeError):
                            continue
            
            if not errors:
                if matching_row is None:
                    errors.append(f"det_id {det_id} not found in results.csv")
                else:
                    # Parse result_row
                    try:
                        class_id = int(matching_row["class_id"]) if matching_row.get("class_id") is not None else None
                    except ValueError:
                        class_id = matching_row["class_id"]
                    
                    try:
                        confidence = float(matching_row["confidence"]) if matching_row.get("confidence") is not None else None
                    except ValueError:
                        confidence = matching_row["confidence"]

                    try:
                        bbox = [
                            float(matching_row["x1"]),
                            float(matching_row["y1"]),
                            float(matching_row["x2"]),
                            float(matching_row["y2"])
                        ]
                    except (ValueError, TypeError, KeyError):
                        bbox = [matching_row.get("x1"), matching_row.get("y1"), matching_row.get("x2"), matching_row.get("y2")]

                    payload["result_row"] = {
                        "eval_id": matching_row.get("eval_id"),
                        "det_id": det_id,
                        "image_id": matching_row.get("image_id"),
                        "source_image": matching_row.get("source_image"),
                        "class_id": class_id,
                        "class_name": matching_row.get("class_name"),
                        "confidence": confidence,
                        "bbox": bbox
                    }

                    # Populate paths
                    image_id = matching_row.get("image_id")
                    source_image = matching_row.get("source_image") or ""
                    ext = os.path.splitext(source_image)[1]

                    payload["source_image_path"] = source_image
                    payload["copied_image_path"] = os.path.normpath(os.path.join(run_dir, "images", f"{image_id}{ext}"))
                    payload["overlay_image_path"] = os.path.normpath(os.path.join(run_dir, "overlays", f"{image_id}_result.jpg"))
                    payload["prediction_json_path"] = os.path.normpath(os.path.join(run_dir, "predictions", f"{image_id}.json"))

        except Exception as e:
            errors.append(f"Failed to read/parse results.csv: {e}")

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    # 4. Load prediction JSON if present, find matching detection
    pred_path = payload["prediction_json_path"]
    if pred_path and os.path.exists(pred_path):
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
                detections = pred_data.get("detections", [])
                for det in detections:
                    if det.get("det_id") == det_id:
                        payload["prediction_detection"] = det
                        break
                if payload["prediction_detection"] is None:
                    warnings.append(f"det_id {det_id} not found in prediction JSON detections list")
        except Exception as e:
            warnings.append(f"Failed to read prediction JSON file {pred_path}: {e}")
    else:
        warnings.append(f"Prediction JSON file not found: {pred_path}")

    # Output
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        r_row = payload["result_row"]
        print("=" * 70)
        print(f" YOLO-Lab Detection Detail (eval_id: {eval_id}, det_id: {det_id})")
        print("=" * 70)
        print(f"Class:           {r_row['class_name']} (ID: {r_row['class_id']})")
        print(f"Confidence:      {r_row['confidence']:.4f}")
        print(f"Bbox:            [{', '.join(f'{x:.2f}' for x in r_row['bbox'])}]")
        
        pred_det = payload.get("prediction_detection")
        if pred_det:
            if "polygon_xy" in pred_det:
                poly = pred_det["polygon_xy"]
                print(f"Polygon:         Yes ({len(poly)} points)")
            if "obb_xywhr" in pred_det:
                obb_xywhr = pred_det["obb_xywhr"]
                print(f"OBB (xywhr):     [{', '.join(f'{x:.4f}' for x in obb_xywhr)}]")
        
        print(f"Image ID:        {r_row['image_id']}")
        print(f"Source Image:    {payload['source_image_path']}")
        print(f"Copied Image:    {payload['copied_image_path']}")
        print(f"Overlay Image:   {payload['overlay_image_path']}")
        print(f"Prediction JSON: {payload['prediction_json_path']}")
        if pred_det:
            print(f"Detection Match: Found in prediction JSON")
        else:
            print(f"Detection Match: Not found or JSON missing")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        print("=" * 70)

def run_mark(args):
    warnings = []
    errors = []

    eval_id = args.eval_id
    det_id = None
    try:
        det_id = int(args.det_id)
    except ValueError:
        errors.append(f"Invalid det_id (must be an integer): {args.det_id}")

    runs_root = args.runs_root or "runs"
    run_dir = os.path.normpath(os.path.join(runs_root, eval_id))
    review_path = os.path.join(run_dir, "review.jsonl")

    # Allowed statuses
    allowed_statuses = {"ok", "missed", "false_positive", "wrong_class", "bad_box", "bad_mask", "low_conf_ok", "unknown"}
    status = args.status
    if status not in allowed_statuses:
        errors.append(f"Invalid status: {status}. Allowed values: {', '.join(sorted(allowed_statuses))}")

    payload = {
        "appended_record": None,
        "review_jsonl_path": os.path.relpath(review_path, os.getcwd()) if os.path.exists(run_dir) else None,
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

    # Validate run dir exists
    if not os.path.exists(run_dir):
        errors.append(f"Run directory does not exist: {run_dir}")
    elif not os.path.isdir(run_dir):
        errors.append(f"Run path is not a directory: {run_dir}")

    # Validate results.csv exists and contains det_id
    csv_path = os.path.join(run_dir, "results.csv")
    if not errors:
        if not os.path.exists(csv_path):
            errors.append(f"results.csv not found in run directory: {run_dir}")
        else:
            det_found = False
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    actual_cols = reader.fieldnames if reader.fieldnames else []
                    if "det_id" not in actual_cols:
                        errors.append(f"results.csv is missing 'det_id' column")
                    else:
                        for row in reader:
                            if not row:
                                continue
                            try:
                                row_det = row.get("det_id")
                                if row_det is not None and int(row_det) == det_id:
                                    det_found = True
                                    break
                            except (ValueError, TypeError):
                                continue
                if not errors and not det_found:
                    errors.append(f"det_id {det_id} not found in results.csv")
            except Exception as e:
                errors.append(f"Failed to read/parse results.csv: {e}")

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    # Append to review.jsonl
    try:
        tz = timezone(timedelta(hours=9)) # JST
        record = {
            "type": "detection",
            "eval_id": eval_id,
            "det_id": det_id,
            "status": status,
            "note": args.note if args.note is not None else "",
            "created_at": datetime.now(tz).isoformat()
        }

        # Open in append mode
        with open(review_path, "a", encoding="utf-8") as rf:
            rf.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        payload["appended_record"] = record
        payload["review_jsonl_path"] = os.path.relpath(review_path, os.getcwd())
    except Exception as e:
        errors.append(f"Failed to write to review.jsonl: {e}")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if errors:
            print("Errors occurred:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("=" * 70)
            print(" YOLO-Lab Review Record Added (detection)")
            print("=" * 70)
            print(f"Eval ID:      {eval_id}")
            print(f"Det ID:       {det_id}")
            print(f"Status:       {status}")
            print(f"Note:         {record['note']}")
            print(f"Created At:   {record['created_at']}")
            print(f"Review Path:  {payload['review_jsonl_path']}")
            print("=" * 70)

def run_mark_image(args):
    warnings = []
    errors = []

    eval_id = args.eval_id
    image_id = args.image_id

    runs_root = args.runs_root or "runs"
    run_dir = os.path.normpath(os.path.join(runs_root, eval_id))
    review_path = os.path.join(run_dir, "review.jsonl")

    # Allowed statuses
    allowed_statuses = {"ok", "missed", "false_positive", "wrong_class", "bad_box", "bad_mask", "low_conf_ok", "unknown"}
    status = args.status
    if status not in allowed_statuses:
        errors.append(f"Invalid status: {status}. Allowed values: {', '.join(sorted(allowed_statuses))}")

    payload = {
        "appended_record": None,
        "review_jsonl_path": os.path.relpath(review_path, os.getcwd()) if os.path.exists(run_dir) else None,
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

    # Validate run dir exists
    if not os.path.exists(run_dir):
        errors.append(f"Run directory does not exist: {run_dir}")
    elif not os.path.isdir(run_dir):
        errors.append(f"Run path is not a directory: {run_dir}")

    # Validate image_id exists
    if not errors:
        image_exists = False

        # 1. Check predictions/<image_id>.json
        pred_json_path = os.path.join(run_dir, "predictions", f"{image_id}.json")
        if os.path.exists(pred_json_path):
            image_exists = True

        # 2. Check copied image under images/
        if not image_exists:
            images_dir = os.path.join(run_dir, "images")
            if os.path.exists(images_dir) and os.path.isdir(images_dir):
                try:
                    for f in os.listdir(images_dir):
                        name, ext = os.path.splitext(f)
                        if name == image_id:
                            image_exists = True
                            break
                except Exception as e:
                    warnings.append(f"Failed to read images directory: {e}")

        # 3. Check results.csv
        if not image_exists:
            csv_path = os.path.join(run_dir, "results.csv")
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        actual_cols = reader.fieldnames if reader.fieldnames else []
                        if "image_id" in actual_cols:
                            for row in reader:
                                if row and row.get("image_id") == image_id:
                                    image_exists = True
                                    break
                except Exception as e:
                    warnings.append(f"Failed to read/parse results.csv: {e}")

        if not image_exists:
            errors.append(f"image_id {image_id} not found in prediction JSON, images directory, or results.csv")

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    # Append to review.jsonl
    try:
        tz = timezone(timedelta(hours=9)) # JST
        record = {
            "type": "image",
            "eval_id": eval_id,
            "image_id": image_id,
            "status": status,
            "target_class": args.target_class if args.target_class is not None else "",
            "note": args.note if args.note is not None else "",
            "created_at": datetime.now(tz).isoformat()
        }

        # Open in append mode
        with open(review_path, "a", encoding="utf-8") as rf:
            rf.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        payload["appended_record"] = record
        payload["review_jsonl_path"] = os.path.relpath(review_path, os.getcwd())
    except Exception as e:
        errors.append(f"Failed to write to review.jsonl: {e}")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if errors:
            print("Errors occurred:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("=" * 70)
            print(" YOLO-Lab Review Record Added (image)")
            print("=" * 70)
            print(f"Eval ID:      {eval_id}")
            print(f"Image ID:     {image_id}")
            print(f"Status:       {status}")
            print(f"Target Class: {record['target_class']}")
            print(f"Note:         {record['note']}")
            print(f"Created At:   {record['created_at']}")
            print(f"Review Path:  {payload['review_jsonl_path']}")
            print("=" * 70)

def compute_summary(eval_id, runs_root):
    warnings = []
    errors = []

    run_dir = os.path.normpath(os.path.join(runs_root, eval_id))

    result = {
        "eval_id": eval_id,
        "run_dir": run_dir,
        "manifest": None,
        "result_counts": {
            "image_count": None,
            "detection_count": None
        },
        "review_counts": {
            "total_records": 0,
            "raw_record_count": 0,
            "active_review_count": 0,
            "by_type": {"detection": 0, "image": 0},
            "by_status": {},
            "by_status_and_type": {"detection": {}, "image": {}}
        },
        "detection_reviews": {},
        "image_reviews": {},
        "problem_candidates": [],
        "warnings": warnings,
        "errors": errors
    }

    if not os.path.exists(run_dir):
        errors.append(f"Run directory does not exist: {run_dir}")
        return result
    elif not os.path.isdir(run_dir):
        errors.append(f"Run path is not a directory: {run_dir}")
        return result

    # 1. Read manifest.json
    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                result["manifest"] = manifest_data
                result["result_counts"]["image_count"] = manifest_data.get("image_count")
                result["result_counts"]["detection_count"] = manifest_data.get("detection_count")
        except Exception as e:
            warnings.append(f"Failed to read manifest.json: {e}")
    else:
        warnings.append("manifest.json not found in run directory")

    # 2. Read results.csv (build map of det_id -> details)
    csv_path = os.path.join(run_dir, "results.csv")
    detections_map = {}
    csv_image_count = 0
    csv_det_count = 0
    if os.path.exists(csv_path):
        try:
            unique_images = set()
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                actual_cols = reader.fieldnames if reader.fieldnames else []
                if "det_id" not in actual_cols:
                    warnings.append("results.csv is missing 'det_id' column")
                else:
                    for row in reader:
                        if not row:
                            continue
                        det_str = row.get("det_id")
                        if det_str is not None:
                            try:
                                det_id = int(det_str)
                            except ValueError:
                                det_id = det_str
                            
                            try:
                                class_id = int(row["class_id"]) if row.get("class_id") is not None else None
                            except ValueError:
                                class_id = row["class_id"]

                            try:
                                confidence = float(row["confidence"]) if row.get("confidence") is not None else None
                            except ValueError:
                                confidence = row["confidence"]

                            try:
                                bbox = [
                                    float(row["x1"]),
                                    float(row["y1"]),
                                    float(row["x2"]),
                                    float(row["y2"])
                                ]
                            except (ValueError, TypeError, KeyError):
                                bbox = [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")]

                            detections_map[str(det_id)] = {
                                "det_id": det_id,
                                "image_id": row.get("image_id"),
                                "source_image": row.get("source_image"),
                                "class_id": class_id,
                                "class_name": row.get("class_name"),
                                "confidence": confidence,
                                "bbox": bbox
                            }
                            csv_det_count += 1
                        img_id = row.get("image_id")
                        if img_id:
                            unique_images.add(img_id)
            csv_image_count = len(unique_images)
            
            # Fill counts if not already set by manifest
            if result["result_counts"]["image_count"] is None:
                result["result_counts"]["image_count"] = csv_image_count
            if result["result_counts"]["detection_count"] is None:
                result["result_counts"]["detection_count"] = csv_det_count
        except Exception as e:
            warnings.append(f"Failed to read/parse results.csv: {e}")
    else:
        warnings.append("results.csv not found in run directory")

    # 3. Read review.jsonl
    review_path = os.path.join(run_dir, "review.jsonl")
    detection_states = {}
    image_states = {}
    total_records = 0

    if os.path.exists(review_path):
        try:
            with open(review_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                        total_records += 1
                        r_type = record.get("type")
                        if r_type == "detection":
                            det_id = str(record.get("det_id"))
                            detection_states[det_id] = record
                        elif r_type == "image":
                            image_id = str(record.get("image_id"))
                            image_states[image_id] = record
                        else:
                            warnings.append(f"review.jsonl line {line_idx}: Unknown record type '{r_type}'")
                    except Exception as e:
                        warnings.append(f"review.jsonl line {line_idx}: Malformed JSON line: {e}")
        except Exception as e:
            warnings.append(f"Failed to read review.jsonl: {e}")
    else:
        warnings.append("review.jsonl not found in run directory")

    # Group and count active review records
    problem_statuses = {"missed", "false_positive", "wrong_class", "bad_box", "bad_mask"}

    # Detection reviews
    det_reviews_out = {}
    for det_id, rec in detection_states.items():
        det_details = detections_map.get(det_id, {})
        det_reviews_out[det_id] = {
            "det_id": rec.get("det_id"),
            "status": rec.get("status"),
            "note": rec.get("note"),
            "created_at": rec.get("created_at"),
            "class_id": det_details.get("class_id"),
            "class_name": det_details.get("class_name"),
            "image_id": det_details.get("image_id"),
            "source_image": det_details.get("source_image")
        }

    # Image reviews
    img_reviews_out = {}
    for image_id, rec in image_states.items():
        img_reviews_out[image_id] = {
            "image_id": rec.get("image_id"),
            "status": rec.get("status"),
            "target_class": rec.get("target_class"),
            "note": rec.get("note"),
            "created_at": rec.get("created_at")
        }

    # Build counts
    active_count = len(detection_states) + len(image_states)
    by_type = {"detection": len(detection_states), "image": len(image_states)}
    by_status = {}
    by_status_and_type = {"detection": {}, "image": {}}

    # Process detections counts
    for det_id, rec in detection_states.items():
        status = rec.get("status")
        if status:
            by_status[status] = by_status.get(status, 0) + 1
            by_status_and_type["detection"][status] = by_status_and_type["detection"].get(status, 0) + 1

    # Process image counts
    for image_id, rec in image_states.items():
        status = rec.get("status")
        if status:
            by_status[status] = by_status.get(status, 0) + 1
            by_status_and_type["image"][status] = by_status_and_type["image"].get(status, 0) + 1

    # Candidates list
    candidates = []
    # Detections candidates
    for det_id, rec in detection_states.items():
        status = rec.get("status")
        if status in problem_statuses:
            det_details = detections_map.get(det_id, {})
            
            image_id = det_details.get("image_id")
            source_image = det_details.get("source_image") or ""
            ext = os.path.splitext(source_image)[1]
            
            copied_image_path = os.path.normpath(os.path.join(run_dir, "images", f"{image_id}{ext}")) if image_id else None
            overlay_image_path = os.path.normpath(os.path.join(run_dir, "overlays", f"{image_id}_result.jpg")) if image_id else None
            prediction_json_path = os.path.normpath(os.path.join(run_dir, "predictions", f"{image_id}.json")) if image_id else None

            candidates.append({
                "type": "detection",
                "eval_id": eval_id,
                "det_id": rec.get("det_id"),
                "image_id": image_id,
                "status": status,
                "note": rec.get("note"),
                "class_id": det_details.get("class_id"),
                "class_name": det_details.get("class_name"),
                "bbox": det_details.get("bbox"),
                "target_class": "",
                "created_at": rec.get("created_at"),
                "source_image": source_image,
                "copied_image_path": copied_image_path,
                "overlay_image_path": overlay_image_path,
                "prediction_json_path": prediction_json_path
            })
    # Image candidates
    for image_id, rec in image_states.items():
        status = rec.get("status")
        if status in problem_statuses:
            pred_json_path = os.path.normpath(os.path.join(run_dir, "predictions", f"{image_id}.json"))
            source_image = ""
            copied_image_path = None
            
            if os.path.exists(pred_json_path):
                try:
                    with open(pred_json_path, "r", encoding="utf-8") as f:
                        pred_data = json.load(f)
                        source_image = pred_data.get("source_image", "")
                except Exception:
                    pass

            if not source_image:
                for det in detections_map.values():
                    if det.get("image_id") == image_id:
                        source_image = det.get("source_image", "")
                        break
            
            ext = os.path.splitext(source_image)[1] if source_image else ".jpg"
            if not source_image:
                images_dir = os.path.join(run_dir, "images")
                if os.path.exists(images_dir) and os.path.isdir(images_dir):
                    try:
                        for f in os.listdir(images_dir):
                            name, f_ext = os.path.splitext(f)
                            if name == image_id:
                                ext = f_ext
                                copied_image_path = os.path.normpath(os.path.join(images_dir, f))
                                break
                    except Exception:
                        pass
            
            if copied_image_path is None:
                copied_image_path = os.path.normpath(os.path.join(run_dir, "images", f"{image_id}{ext}"))

            overlay_image_path = os.path.normpath(os.path.join(run_dir, "overlays", f"{image_id}_result.jpg"))
            prediction_json_path = pred_json_path

            candidates.append({
                "type": "image",
                "eval_id": eval_id,
                "det_id": None,
                "image_id": image_id,
                "status": status,
                "note": rec.get("note"),
                "class_id": None,
                "class_name": None,
                "bbox": None,
                "target_class": rec.get("target_class"),
                "created_at": rec.get("created_at"),
                "source_image": source_image,
                "copied_image_path": copied_image_path,
                "overlay_image_path": overlay_image_path,
                "prediction_json_path": prediction_json_path
            })

    # Fill result
    result["review_counts"]["total_records"] = total_records
    result["review_counts"]["raw_record_count"] = total_records
    result["review_counts"]["active_review_count"] = active_count
    result["review_counts"]["by_type"] = by_type
    result["review_counts"]["by_status"] = by_status
    result["review_counts"]["by_status_and_type"] = by_status_and_type
    result["detection_reviews"] = det_reviews_out
    result["image_reviews"] = img_reviews_out
    result["problem_candidates"] = candidates

    return result

def run_summary(args):
    runs_root = args.runs_root or "runs"
    payload = compute_summary(args.eval_id, runs_root)

    if payload["errors"]:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in payload["errors"]:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(f" YOLO-Lab Evaluation Run Review Summary (eval_id: {payload['eval_id']})")
        print("=" * 70)
        print(f"Raw Records Count:     {payload['review_counts']['raw_record_count']}")
        print(f"Active Reviews Count:  {payload['review_counts']['active_review_count']}")
        print(f"Active Detections:     {payload['review_counts']['by_type']['detection']}")
        print(f"Active Images:         {payload['review_counts']['by_type']['image']}")
        print("-" * 70)
        print("Counts by Status (Active):")
        by_status = payload['review_counts']['by_status']
        if by_status:
            for status, count in sorted(by_status.items()):
                print(f"  - {status}: {count}")
        else:
            print("  (No reviews registered yet)")
        print("-" * 70)
        print(f"Problem Candidates:    {len(payload['problem_candidates'])}")
        if payload["warnings"]:
            print("\nWarnings:")
            for w in payload["warnings"]:
                print(f"  - {w}")
        print("=" * 70)

def run_export_candidates(args):
    warnings = []
    errors = []

    eval_id = args.eval_id
    runs_root = args.runs_root or "runs"
    run_dir = os.path.normpath(os.path.join(runs_root, eval_id))

    allowed_statuses_list = {"ok", "missed", "false_positive", "wrong_class", "bad_box", "bad_mask", "low_conf_ok", "unknown"}

    if args.status:
        selected_statuses = [s.strip() for s in args.status.split(",") if s.strip()]
        for s in selected_statuses:
            if s not in allowed_statuses_list:
                errors.append(f"Invalid status in filter: {s}. Allowed values: {', '.join(sorted(allowed_statuses_list))}")
    else:
        selected_statuses = ["missed", "false_positive", "wrong_class", "bad_box", "bad_mask"]

    payload = {
        "candidate_count": 0,
        "candidates_json_path": None,
        "candidates_csv_path": None,
        "images_txt_path": None,
        "selected_statuses": selected_statuses,
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

    summary_data = compute_summary(eval_id, runs_root)
    errors.extend(summary_data["errors"])
    warnings.extend(summary_data["warnings"])

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    if args.out:
        out_dir = os.path.abspath(args.out)
    else:
        out_dir = os.path.join(run_dir, "candidates")

    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        errors.append(f"Failed to create output directory {out_dir}: {e}")
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(0)
        else:
            print(f"Failed to create output directory {out_dir}: {e}", file=sys.stderr)
            sys.exit(1)

    raw_candidates = summary_data["problem_candidates"]
    filtered_candidates = []
    for cand in raw_candidates:
        if cand.get("status") in selected_statuses:
            filtered_candidates.append(cand)

    json_out_path = os.path.join(out_dir, "candidates.json")
    csv_out_path = os.path.join(out_dir, "candidates.csv")
    txt_out_path = os.path.join(out_dir, "images.txt")

    try:
        tz = timezone(timedelta(hours=9)) # JST
        export_json_data = {
            "eval_id": eval_id,
            "run_dir": run_dir,
            "selected_statuses": selected_statuses,
            "candidate_count": len(filtered_candidates),
            "candidates": filtered_candidates,
            "generated_at": datetime.now(tz).isoformat()
        }
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(export_json_data, f, indent=2, ensure_ascii=False)
        payload["candidates_json_path"] = os.path.relpath(json_out_path, os.getcwd())
    except Exception as e:
        errors.append(f"Failed to write candidates.json: {e}")

    try:
        with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "eval_id", "type", "status", "det_id", "image_id", 
                "source_image", "copied_image_path", "overlay_image_path", "prediction_json_path",
                "class_id", "class_name", "target_class", "bbox", "note", "created_at"
            ])
            for cand in filtered_candidates:
                bbox_str = json.dumps(cand.get("bbox")) if cand.get("bbox") is not None else ""
                writer.writerow([
                    cand.get("eval_id"),
                    cand.get("type"),
                    cand.get("status"),
                    cand.get("det_id"),
                    cand.get("image_id"),
                    cand.get("source_image"),
                    cand.get("copied_image_path"),
                    cand.get("overlay_image_path"),
                    cand.get("prediction_json_path"),
                    cand.get("class_id"),
                    cand.get("class_name"),
                    cand.get("target_class"),
                    bbox_str,
                    cand.get("note"),
                    cand.get("created_at")
                ])
        payload["candidates_csv_path"] = os.path.relpath(csv_out_path, os.getcwd())
    except Exception as e:
        errors.append(f"Failed to write candidates.csv: {e}")

    try:
        unique_image_paths = []
        for cand in filtered_candidates:
            path_to_use = cand.get("copied_image_path")
            if not path_to_use:
                path_to_use = cand.get("source_image")
            if path_to_use and path_to_use not in unique_image_paths:
                unique_image_paths.append(path_to_use)

        with open(txt_out_path, "w", encoding="utf-8") as f:
            for path in unique_image_paths:
                f.write(path + "\n")
        payload["images_txt_path"] = os.path.relpath(txt_out_path, os.getcwd())
    except Exception as e:
        errors.append(f"Failed to write images.txt: {e}")

    payload["candidate_count"] = len(filtered_candidates)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if errors:
            print("Errors occurred:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
        else:
            print("=" * 70)
            print(f" YOLO-Lab Candidates Exported (eval_id: {eval_id})")
            print("=" * 70)
            print(f"Candidate Count:      {payload['candidate_count']}")
            print(f"Candidates JSON:      {payload['candidates_json_path']}")
            print(f"Candidates CSV:       {payload['candidates_csv_path']}")
            print(f"Unique Images List:   {payload['images_txt_path']}")
            print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="AI-first Evaluation and Review CLI for YOLO-Lab")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # evaluate subcommand
    parser_eval = subparsers.add_parser("evaluate", help="Run model evaluation on an image folder")
    parser_eval.add_argument("--model", type=str, required=True, help="Path to PT/Ultralytics model")
    parser_eval.add_argument("--input", type=str, required=True, help="Path to input image directory")
    parser_eval.add_argument("--out", type=str, required=True, help="Path to output run directory")
    parser_eval.add_argument("--conf", type=float, help="Optional confidence threshold")
    parser_eval.add_argument("--imgsz", type=int, help="Optional inference image size")
    parser_eval.add_argument("--geometry", type=str, choices=["auto", "bbox", "polygon", "obb"], default="auto", help="Output geometry type")
    parser_eval.add_argument("--overlay", type=str, choices=["auto", "bbox", "polygon", "mask", "obb", "both"], default="auto", help="Overlay drawing style")
    parser_eval.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    # show subcommand
    parser_show = subparsers.add_parser("show", help="Show detection details by eval_id and det_id")
    parser_show.add_argument("eval_id", type=str, help="Evaluation run ID")
    parser_show.add_argument("det_id", type=str, help="Detection ID")
    parser_show.add_argument("--runs-root", type=str, default="runs", help="Root directory for runs")
    parser_show.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    # mark subcommand
    parser_mark = subparsers.add_parser("mark", help="Append a detection-level review record")
    parser_mark.add_argument("eval_id", type=str, help="Evaluation run ID")
    parser_mark.add_argument("det_id", type=str, help="Detection ID")
    parser_mark.add_argument("--status", type=str, required=True, help="Review status")
    parser_mark.add_argument("--note", type=str, help="Optional text note")
    parser_mark.add_argument("--runs-root", type=str, default="runs", help="Root directory for runs")
    parser_mark.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    # mark-image subcommand
    parser_mark_image = subparsers.add_parser("mark-image", help="Append an image-level review record")
    parser_mark_image.add_argument("eval_id", type=str, help="Evaluation run ID")
    parser_mark_image.add_argument("image_id", type=str, help="Image ID (e.g. 000001)")
    parser_mark_image.add_argument("--status", type=str, required=True, help="Review status")
    parser_mark_image.add_argument("--target-class", type=str, help="Optional target class for missed/wrong class")
    parser_mark_image.add_argument("--note", type=str, help="Optional text note")
    parser_mark_image.add_argument("--runs-root", type=str, default="runs", help="Root directory for runs")
    parser_mark_image.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    # summary subcommand
    parser_summary = subparsers.add_parser("summary", help="Summarize review records for an evaluation run")
    parser_summary.add_argument("eval_id", type=str, help="Evaluation run ID")
    parser_summary.add_argument("--runs-root", type=str, default="runs", help="Root directory for runs")
    parser_summary.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    # export-candidates subcommand
    parser_export = subparsers.add_parser("export-candidates", help="Export review candidates for dataset editing/fine-tuning")
    parser_export.add_argument("eval_id", type=str, help="Evaluation run ID")
    parser_export.add_argument("--runs-root", type=str, default="runs", help="Root directory for runs")
    parser_export.add_argument("--out", type=str, help="Output directory path (default: <run>/candidates)")
    parser_export.add_argument("--status", type=str, help="Comma-separated review statuses to export")
    parser_export.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")

    args = parser.parse_args()

    if args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "show":
        run_show(args)
    elif args.command == "mark":
        run_mark(args)
    elif args.command == "mark-image":
        run_mark_image(args)
    elif args.command == "summary":
        run_summary(args)
    elif args.command == "export-candidates":
        run_export_candidates(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
