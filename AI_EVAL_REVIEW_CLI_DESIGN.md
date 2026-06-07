# AI Eval Review CLI Design

Last updated: 2026-06-06

## Purpose

`ai-eval` is an AI-first evaluation and review CLI.

Its purpose is not to be a GUI or an annotation application. Its purpose is to turn image-folder inference results into stable IDs, structured files, review records, and later fine-tune candidates that a generative AI can use efficiently.

The core workflow is:

```text
evaluate images -> assign stable detection IDs -> review by ID -> summarize review -> export candidates for dataset editing / fine-tune
```

## Position In YOLO-Lab

Existing probes answer local technical questions:

```text
ai-env-probe
ai-model-probe
ai-image-predict-probe
ai-val-probe
ai-onnx-probe
ai-onnx-raw-inference-probe
ai-compare-probe
```

`ai-eval` is different. It manages a reviewable evaluation run.

It should become the bridge between:

```text
model prediction
human review
dataset editing
light fine-tuning
```

## ID Model

### eval_id

An evaluation run ID.

Example:

```text
id0001
id0002
id0003
```

Default output directory:

```text
runs/id0001/
```

The first implementation may require `--out runs/id0001` and derive `eval_id` from the folder name. Automatic ID allocation can come later.

### image_id

A stable image number within the evaluation.

Example:

```text
000001
000002
000003
```

Images are assigned in deterministic sorted order.

### det_id

A stable detection number across the whole evaluation run.

Example:

```text
image 000001: det_id 1, 2
image 000002: det_id 3
image 000003: det_id 4, 5, 6
```

The user should mostly refer to detections as:

```text
id0001 505
```

Meaning:

```text
eval_id = id0001
det_id = 505
```

## Output Structure

Preferred structure:

```text
runs/id0001/
  manifest.json
  results.csv
  review.jsonl
  images/
    000001.jpg
    000002.jpg
  overlays/
    000001_result.jpg
    000002_result.jpg
  predictions/
    000001.json
    000002.json
```

### manifest.json

Run-level metadata.

Minimum fields:

```json
{
  "eval_id": "id0001",
  "created_at": "2026-06-06T00:00:00+09:00",
  "model": "models/standard/yolov8s.pt",
  "input": "samples/images",
  "task": "detect",
  "conf": 0.25,
  "imgsz": 640,
  "image_count": 2,
  "detection_count": 11
}
```

### results.csv

Detection index.

Minimum columns:

```text
eval_id,det_id,image_id,source_image,class_id,class_name,confidence,x1,y1,x2,y2
```

Example:

```text
id0001,505,000238,input/B.jpg,28,suitcase,0.63,120,80,450,500
id0001,506,000238,input/B.jpg,72,refrigerator,0.41,500,100,800,620
```

For instance-level geometry (segmentation polygon, OBB), add rich geometry directly to `predictions/*.json` instead of overloading the first version of the CSV. The CSV maintains backward compatibility with bounding boxes as the primary stable instance index.

### predictions/*.json

Image-level prediction file. Stores rich instance geometry like `polygon_xy`, `polygon_xyn`, `obb_xywhr`, and `obb_xyxyxyxy`.

Current geometry support is instance-level: detect, segment, and OBB. Classification and semantic segmentation are future planned output families because they are image-level or otherwise different from instance geometry; they are intentionally outside the current implementation scope.

Example:

```json
{
  "eval_id": "id0001",
  "image_id": "000238",
  "source_image": "input/B.jpg",
  "overlay_image": "overlays/000238_result.jpg",
  "detections": [
    {
      "det_id": 505,
      "class_id": 28,
      "class_name": "suitcase",
      "confidence": 0.63,
      "bbox_xyxy": [120, 80, 450, 500]
    }
  ]
}
```

### review.jsonl

Append-only review records.

Detection-level record:

```json
{"type":"detection","eval_id":"id0001","det_id":506,"status":"false_positive","note":"冷蔵庫誤検出"}
```

Image-level record:

```json
{"type":"image","eval_id":"id0001","image_id":"000238","status":"missed","target_class":"suitcase","note":"右下のスーツケース未検出"}
```

## Overlay Drawing

Overlay images should be usable in an ordinary image viewer.

Minimum drawing:

```text
eval: id0001 image: 000238
[505] suitcase 0.63
[506] refrigerator 0.41
```

Each detection box should include:

```text
[det_id] class_name confidence
```

This enables natural language review:

```text
id0001の505はOK
id0001の506は誤検出
```

## Commands

### evaluate

Run inference for an image folder and create a reviewable evaluation run.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py evaluate --model .\models\standard\yolov8s.pt --input .\samples\images --out .\runs\id0001 --json
```

Initial scope:

```text
PT/Ultralytics model only
image folder input
detect boxes only, with room for segment later
manifest.json
results.csv
predictions/*.json
overlays/*.jpg
empty review.jsonl
```

ONNX models can be added later after decoded ONNX output support exists.

### show

Show detection details by `eval_id` and `det_id`.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py show id0001 505
```

Output should include:

```text
eval_id
det_id
image_id
source_image
class
confidence
bbox
overlay path
```

### mark

Append a detection-level review record.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py mark id0001 506 --status false_positive --note "冷蔵庫誤検出"
```

### mark-image

Append an image-level review record, used for missed detections and other image-level notes.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py mark-image id0001 000238 --status missed --target-class suitcase --note "右下のスーツケース未検出"
```

### summary

Summarize review records.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py summary id0001
```

### export-candidates

Export images/detections for later dataset editing or fine-tune preparation.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py export-candidates id0001 --status missed,wrong_class,false_positive,bad_box,bad_mask
```

This should be implemented later, after `mark` and `mark-image`.

## Review Status

Allowed statuses:

```text
ok
missed
false_positive
wrong_class
bad_box
bad_mask
low_conf_ok
unknown
```

Meaning:

```text
ok: detection is correct
missed: target object was not detected
false_positive: detection should not exist
wrong_class: object exists, but class is wrong
bad_box: detection class is right but bbox is poor
bad_mask: segmentation mask is poor
low_conf_ok: confidence is low, but detection is correct
unknown: human review is undecided
```

## Implementation Order

Recommended implementation order:

```text
1. ai-eval evaluate
2. ai-eval show
3. ai-eval mark
4. ai-eval mark-image
5. ai-eval summary
6. ai-eval export-candidates
```

The first implementation should only do `evaluate`.

## Non-Goals For First Implementation

Do not implement in the first version:

```text
ONNX decoded prediction
segmentation masks in CSV
OBB
interactive GUI
automatic fine-tuning
format conversion
review editing/deleting
database storage
```

## Design Principle

`ai-eval` should make human review addressable by stable IDs.

The CLI manages:

```text
evaluation
ID assignment
review records
candidate extraction
```

The generative AI translates natural language into CLI commands.

The human mostly views images and says what should be marked.
