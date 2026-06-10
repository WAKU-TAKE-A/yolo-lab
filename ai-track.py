import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_classes_arg(classes_arg):
    if not classes_arg:
        return None, None
    allowed_ids = set()
    allowed_names = set()
    for part in classes_arg.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            allowed_ids.add(int(item))
        except ValueError:
            allowed_names.add(item)
    return allowed_ids, allowed_names


def parse_roi_arg(roi_arg):
    if roi_arg is None:
        return None
    parts = [float(x.strip()) for x in roi_arg.split(",")]
    if len(parts) != 4:
        raise ValueError(f"roi requires exactly 4 numbers, got {len(parts)}")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("roi must satisfy x2 > x1 and y2 > y1")
    return parts


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def bbox_center_in_roi(bbox, roi):
    if roi is None:
        return True
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_track(args):
    warnings = []
    errors = []

    resolved_model = os.path.abspath(args.model)
    resolved_input = os.path.abspath(args.input)
    resolved_out = os.path.abspath(args.out)
    run_id = os.path.basename(os.path.normpath(resolved_out))

    payload = {
        "run_id": run_id,
        "out_dir": args.out,
        "frame_count": 0,
        "detection_count": 0,
        "track_count": 0,
        "manifest_path": None,
        "detections_jsonl_path": None,
        "tracks_jsonl_path": None,
        "events_jsonl_path": None,
        "track_summary_path": None,
        "overlays_dir": None,
        "timing": {
            "total_track_seconds": 0.0,
            "avg_frame_ms": 0.0,
            "frames_per_second": 0.0,
            "total_predict_ms": 0.0,
            "avg_predict_ms": 0.0,
        },
        "warnings": warnings,
        "errors": errors,
    }

    if not os.path.exists(resolved_model):
        errors.append(f"Model file does not exist: {args.model}")
    if not os.path.exists(resolved_input):
        errors.append(f"Input directory does not exist: {args.input}")
    elif not os.path.isdir(resolved_input):
        errors.append(f"Input path is not a directory: {args.input}")
    if args.fps <= 0:
        errors.append("--fps must be greater than 0")
    if args.save_overlays not in {"all", "none"}:
        errors.append(f"Unsupported --save-overlays value: {args.save_overlays}")
    if args.tracker != "simple":
        errors.append(f"Unsupported --tracker value: {args.tracker}")
    if args.iou_threshold < 0.0 or args.iou_threshold > 1.0:
        errors.append("--iou-threshold must be between 0 and 1")
    if args.max_missing_frames < 0:
        errors.append("--max-missing-frames must be 0 or greater")
    if args.min_box_width is not None and args.min_box_width < 0:
        errors.append("--min-box-width must be 0 or greater")
    if args.min_box_height is not None and args.min_box_height < 0:
        errors.append("--min-box-height must be 0 or greater")
    if args.min_box_area is not None and args.min_box_area < 0:
        errors.append("--min-box-area must be 0 or greater")

    allowed_ids = None
    allowed_names = None
    roi = None
    if not errors:
        try:
            allowed_ids, allowed_names = parse_classes_arg(args.classes)
        except Exception as e:
            errors.append(f"Failed to parse --classes: {e}")
        try:
            roi = parse_roi_arg(args.roi)
        except Exception as e:
            errors.append(f"Failed to parse --roi: {e}")

    if os.path.exists(resolved_out):
        if not args.force:
            errors.append(f"Output directory already exists: {resolved_out}. Use --force to overwrite.")
        else:
            try:
                shutil.rmtree(resolved_out)
            except Exception as e:
                errors.append(f"Failed to remove existing output directory: {e}")

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(1)
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)

    image_files = []
    try:
        for name in os.listdir(resolved_input):
            ext = os.path.splitext(name)[1].lower()
            if ext in VALID_IMAGE_EXTS:
                image_files.append(name)
        image_files.sort()
    except Exception as e:
        errors.append(f"Failed to scan input directory: {e}")

    if not image_files:
        warnings.append(f"No valid image files found in input directory: {args.input}")

    if errors:
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(1)
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)

    try:
        import cv2
        from ultralytics import YOLO
    except Exception as e:
        errors.append(f"Failed to import required libraries: {e}")
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(1)
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)

    try:
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
    except Exception:
        pass

    os.makedirs(resolved_out, exist_ok=True)
    overlays_dir = os.path.join(resolved_out, "overlays")
    if args.save_overlays == "all":
        os.makedirs(overlays_dir, exist_ok=True)

    detections_jsonl_path = os.path.join(resolved_out, "detections.jsonl")
    tracks_jsonl_path = os.path.join(resolved_out, "tracks.jsonl")
    events_jsonl_path = os.path.join(resolved_out, "events.jsonl")
    track_summary_path = os.path.join(resolved_out, "track_summary.json")
    manifest_path = os.path.join(resolved_out, "manifest.json")
    for jsonl_path in [detections_jsonl_path, tracks_jsonl_path, events_jsonl_path]:
        with open(jsonl_path, "w", encoding="utf-8"):
            pass

    f_stdout = io.StringIO()
    f_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
            model = YOLO(resolved_model)
    except Exception as e:
        errors.append(f"Failed to load YOLO model: {e}")
        if args.json:
            print(json.dumps(payload, indent=2))
            sys.exit(1)
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)

    next_det_id = 1
    next_track_id = 1
    active_tracks = {}
    ended_tracks = []
    total_predict_ms = 0.0
    run_started = time.perf_counter()

    for frame_index, fname in enumerate(image_files):
        frame_id = f"{frame_index + 1:06d}"
        time_sec = frame_index / float(args.fps)
        src_img_path = os.path.join(resolved_input, fname)
        source_image = os.path.relpath(src_img_path, os.getcwd())

        predict_kwargs = {"source": src_img_path, "save": False, "verbose": False}
        if args.conf is not None:
            predict_kwargs["conf"] = args.conf
        if args.imgsz is not None:
            predict_kwargs["imgsz"] = args.imgsz

        frame_detections = []
        try:
            predict_started = time.perf_counter()
            with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
                results = model.predict(**predict_kwargs)
            predict_ms = (time.perf_counter() - predict_started) * 1000.0
            total_predict_ms += predict_ms
            result = results[0]
        except Exception as e:
            warnings.append(f"Prediction failed on frame {fname}: {e}")
            continue

        boxes = None
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
        elif hasattr(result, "obb") and result.obb is not None:
            boxes = result.obb

        if boxes is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names[cls_id] if result.names and cls_id in result.names else f"class_{cls_id}"
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().tolist()
                bbox_width = float(xyxy[2] - xyxy[0])
                bbox_height = float(xyxy[3] - xyxy[1])
                bbox_area = float(max(0.0, bbox_width) * max(0.0, bbox_height))

                if allowed_ids is not None or allowed_names is not None:
                    id_ok = allowed_ids is not None and cls_id in allowed_ids
                    name_ok = allowed_names is not None and cls_name in allowed_names
                    if not id_ok and not name_ok:
                        continue
                if args.min_box_width is not None and bbox_width < args.min_box_width:
                    continue
                if args.min_box_height is not None and bbox_height < args.min_box_height:
                    continue
                if args.min_box_area is not None and bbox_area < args.min_box_area:
                    continue
                if not bbox_center_in_roi(xyxy, roi):
                    continue

                det_record = {
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "time_sec": time_sec,
                    "source_image": source_image,
                    "det_id": next_det_id,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf_val,
                    "bbox_xyxy": xyxy,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_area": bbox_area,
                }
                frame_detections.append(det_record)
                append_jsonl(detections_jsonl_path, det_record)
                next_det_id += 1

        matches = []
        if active_tracks and frame_detections:
            pairs = []
            for track_id, track in active_tracks.items():
                for det_idx, det in enumerate(frame_detections):
                    if track["class_id"] != det["class_id"]:
                        continue
                    iou = bbox_iou(track["last_bbox"], det["bbox_xyxy"])
                    if iou >= args.iou_threshold:
                        pairs.append((iou, track_id, det_idx))
            pairs.sort(reverse=True)
            used_tracks = set()
            used_dets = set()
            for iou, track_id, det_idx in pairs:
                if track_id in used_tracks or det_idx in used_dets:
                    continue
                used_tracks.add(track_id)
                used_dets.add(det_idx)
                matches.append((track_id, det_idx, iou))

        matched_track_ids = set()
        matched_det_indexes = set()
        new_track_ids = set()
        for track_id, det_idx, _ in matches:
            matched_track_ids.add(track_id)
            matched_det_indexes.add(det_idx)
            track = active_tracks[track_id]
            det = frame_detections[det_idx]
            if track["state"] == "lost":
                append_jsonl(events_jsonl_path, {
                    "event": "track_resumed",
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "time_sec": time_sec,
                })
            track["state"] = "active"
            track["missing_frames"] = 0
            track["last_bbox"] = det["bbox_xyxy"]
            track["last_frame"] = frame_id
            track["last_frame_index"] = frame_index
            track["last_time_sec"] = time_sec
            track["matched_frame_count"] += 1
            track["max_confidence"] = max(track["max_confidence"], det["confidence"])
            track["max_bbox_area"] = max(track["max_bbox_area"], det["bbox_area"])

            track_record = {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "time_sec": time_sec,
                "track_id": track_id,
                "det_id": det["det_id"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox_xyxy": det["bbox_xyxy"],
                "state": "active",
                "counted": False,
            }
            append_jsonl(tracks_jsonl_path, track_record)

        for det_idx, det in enumerate(frame_detections):
            if det_idx in matched_det_indexes:
                continue
            track_id = next_track_id
            next_track_id += 1
            new_track_ids.add(track_id)
            active_tracks[track_id] = {
                "track_id": track_id,
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "first_frame": frame_id,
                "last_frame": frame_id,
                "first_frame_index": frame_index,
                "last_frame_index": frame_index,
                "first_time_sec": time_sec,
                "last_time_sec": time_sec,
                "matched_frame_count": 1,
                "max_confidence": det["confidence"],
                "max_bbox_area": det["bbox_area"],
                "missing_frames": 0,
                "state": "active",
                "last_bbox": det["bbox_xyxy"],
            }
            append_jsonl(events_jsonl_path, {
                "event": "track_started",
                "track_id": track_id,
                "frame_id": frame_id,
                "frame_index": frame_index,
                "time_sec": time_sec,
            })
            append_jsonl(tracks_jsonl_path, {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "time_sec": time_sec,
                "track_id": track_id,
                "det_id": det["det_id"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox_xyxy": det["bbox_xyxy"],
                "state": "active",
                "counted": False,
            })

        ended_now = []
        for track_id, track in active_tracks.items():
            if track_id in matched_track_ids:
                continue
            if track_id in new_track_ids:
                continue
            track["missing_frames"] += 1
            if track["missing_frames"] == 1 and track["state"] != "lost":
                track["state"] = "lost"
                append_jsonl(events_jsonl_path, {
                    "event": "track_lost",
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "time_sec": time_sec,
                })
            if track["missing_frames"] > args.max_missing_frames:
                track["state"] = "ended"
                append_jsonl(events_jsonl_path, {
                    "event": "track_ended",
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "time_sec": time_sec,
                })
                ended_now.append(track_id)

        for track_id in ended_now:
            ended_tracks.append(active_tracks.pop(track_id))

        if args.save_overlays == "all":
            try:
                img = cv2.imread(src_img_path)
                if img is None:
                    warnings.append(f"Failed to read frame for overlay: {src_img_path}")
                else:
                    text_run = f"track: {run_id} frame: {frame_id} t={time_sec:.2f}s"
                    cv2.putText(img, text_run, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, text_run, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    matched_by_det_id = {}
                    for track_id, det_idx, _ in matches:
                        matched_by_det_id[frame_detections[det_idx]["det_id"]] = track_id
                    for det_idx, det in enumerate(frame_detections):
                        if det["det_id"] in matched_by_det_id:
                            track_id = matched_by_det_id[det["det_id"]]
                        else:
                            track_id = None
                            for candidate_track_id, track in active_tracks.items():
                                if (
                                    track["last_frame"] == frame_id
                                    and track["class_id"] == det["class_id"]
                                    and track["last_bbox"] == det["bbox_xyxy"]
                                ):
                                    track_id = candidate_track_id
                                    break
                        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"T{track_id} [{det['det_id']}] {det['class_name']} {det['confidence']:.2f}" if track_id else f"[{det['det_id']}] {det['class_name']} {det['confidence']:.2f}"
                        (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        y_text = y1 - 4
                        if y_text - lbl_h < 0:
                            cv2.rectangle(img, (x1, y1), (x1 + lbl_w + 4, y1 + lbl_h + 4), (0, 255, 0), -1)
                            cv2.putText(img, label, (x1 + 2, y1 + lbl_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        else:
                            cv2.rectangle(img, (x1, y1 - lbl_h - 4), (x1 + lbl_w + 4, y1), (0, 255, 0), -1)
                            cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.imwrite(os.path.join(overlays_dir, f"{frame_id}_track.jpg"), img)
            except Exception as e:
                warnings.append(f"Failed to write overlay for frame {fname}: {e}")

        payload["frame_count"] = frame_index + 1
        payload["detection_count"] += len(frame_detections)

    for track_id, track in list(active_tracks.items()):
        if track["state"] != "ended":
            append_jsonl(events_jsonl_path, {
                "event": "track_ended",
                "track_id": track_id,
                "frame_id": track["last_frame"],
                "frame_index": track["last_frame_index"],
                "time_sec": track["last_time_sec"],
            })
            track["state"] = "ended"
            ended_tracks.append(track)
    active_tracks.clear()

    total_track_seconds = time.perf_counter() - run_started
    frame_count = payload["frame_count"]
    track_count = len(ended_tracks)
    payload["track_count"] = track_count
    payload["timing"] = {
        "total_track_seconds": total_track_seconds,
        "avg_frame_ms": (total_track_seconds * 1000.0 / frame_count) if frame_count > 0 else 0.0,
        "frames_per_second": (frame_count / total_track_seconds) if total_track_seconds > 0 else 0.0,
        "total_predict_ms": total_predict_ms,
        "avg_predict_ms": (total_predict_ms / frame_count) if frame_count > 0 else 0.0,
    }

    track_summaries = []
    for track in sorted(ended_tracks, key=lambda item: item["track_id"]):
        track_summaries.append({
            "track_id": track["track_id"],
            "class_id": track["class_id"],
            "class_name": track["class_name"],
            "first_frame": track["first_frame"],
            "last_frame": track["last_frame"],
            "first_frame_index": track["first_frame_index"],
            "last_frame_index": track["last_frame_index"],
            "first_time_sec": track["first_time_sec"],
            "last_time_sec": track["last_time_sec"],
            "duration_sec": track["last_time_sec"] - track["first_time_sec"],
            "matched_frame_count": track["matched_frame_count"],
            "max_confidence": track["max_confidence"],
            "max_bbox_area": track["max_bbox_area"],
            "counted": False,
        })

    tz = timezone(timedelta(hours=9))
    manifest_data = {
        "run_id": run_id,
        "created_at": datetime.now(tz).isoformat(),
        "input": args.input,
        "model": args.model,
        "fps": float(args.fps),
        "frame_count": frame_count,
        "detection_count": payload["detection_count"],
        "track_count": track_count,
        "conf": args.conf if args.conf is not None else 0.25,
        "imgsz": args.imgsz,
        "save_overlays": args.save_overlays,
        "classes": [item.strip() for item in args.classes.split(",") if item.strip()] if args.classes else [],
        "filters": {
            "min_box_width": args.min_box_width,
            "min_box_height": args.min_box_height,
            "min_box_area": args.min_box_area,
            "roi": roi,
        },
        "tracker": {
            "type": args.tracker,
            "iou_threshold": args.iou_threshold,
            "max_missing_frames": args.max_missing_frames,
        },
        "timing": payload["timing"],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    track_summary = {
        "run_id": run_id,
        "frame_count": frame_count,
        "detection_count": payload["detection_count"],
        "track_count": track_count,
        "tracks": track_summaries,
    }
    with open(track_summary_path, "w", encoding="utf-8") as f:
        json.dump(track_summary, f, indent=2)

    payload["manifest_path"] = os.path.relpath(manifest_path, os.getcwd())
    payload["detections_jsonl_path"] = os.path.relpath(detections_jsonl_path, os.getcwd())
    payload["tracks_jsonl_path"] = os.path.relpath(tracks_jsonl_path, os.getcwd())
    payload["events_jsonl_path"] = os.path.relpath(events_jsonl_path, os.getcwd())
    payload["track_summary_path"] = os.path.relpath(track_summary_path, os.getcwd())
    payload["overlays_dir"] = os.path.relpath(overlays_dir, os.getcwd()) if args.save_overlays == "all" else None

    captured_stdout = f_stdout.getvalue().strip()
    captured_stderr = f_stderr.getvalue().strip()
    if captured_stdout:
        warnings.append(f"Captured stdout during tracking: {captured_stdout}")
    if captured_stderr:
        warnings.append(f"Captured stderr during tracking: {captured_stderr}")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 70)
        print(" YOLO-Lab Tracking Summary")
        print("=" * 70)
        print(f"Run ID:          {payload['run_id']}")
        print(f"Output Run Dir:  {payload['out_dir']}")
        print(f"Frame Count:     {payload['frame_count']}")
        print(f"Detections:      {payload['detection_count']}")
        print(f"Track Count:     {payload['track_count']}")
        print(f"Manifest Path:   {payload['manifest_path']}")
        print(f"Tracks JSONL:    {payload['tracks_jsonl_path']}")
        if payload["overlays_dir"]:
            print(f"Overlays Dir:    {payload['overlays_dir']}")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  - {warning}")
        if errors:
            print("\nErrors:")
            for error in errors:
                print(f"  - {error}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AI-first Image Sequence Tracking CLI for YOLO-Lab")
    parser.add_argument("--model", required=True, help="Path to PT/Ultralytics model")
    parser.add_argument("--input", required=True, help="Path to input image directory")
    parser.add_argument("--fps", required=True, type=float, help="Frame rate represented by the image sequence")
    parser.add_argument("--out", required=True, help="Path to output run directory")
    parser.add_argument("--conf", type=float, help="Optional confidence threshold")
    parser.add_argument("--imgsz", type=int, help="Optional inference image size")
    parser.add_argument("--classes", help="Comma-separated class names or IDs to keep")
    parser.add_argument("--min-box-width", type=float, help="Minimum bbox width in pixels")
    parser.add_argument("--min-box-height", type=float, help="Minimum bbox height in pixels")
    parser.add_argument("--min-box-area", type=float, help="Minimum bbox area in pixels")
    parser.add_argument("--roi", help="Optional ROI x1,y1,x2,y2")
    parser.add_argument("--save-overlays", choices=["all", "none"], default="all", help="Overlay saving mode")
    parser.add_argument("--tracker", choices=["simple"], default="simple", help="Tracker type")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for track matching")
    parser.add_argument("--max-missing-frames", type=int, default=5, help="Frames to keep unmatched tracks alive")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")
    args = parser.parse_args()
    run_track(args)


if __name__ == "__main__":
    main()
