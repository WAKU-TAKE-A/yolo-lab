# YOLO-Lab README for AI Agents

This repository provides an AI-first command workflow for YOLO, ONNXRuntime, evaluation review, dataset conversion, and light fine-tuning. The purpose is not to build a GUI or a general training platform. The purpose is to give an AI assistant stable commands that return structured facts, usually JSON, so the assistant can answer YOLO-related questions with less drift.

Public repository positioning: keep the repository name and scope centered on YOLO-Lab. This is an AI-friendly YOLO toolkit, not a generic AI lab.

Use this file when you are an AI agent operating this repository.

## Core Principle

Prefer standard models first. Fine-tuning is a later step, not the default starting point.

The intended loop is:

```text
environment/model check
-> image-level prediction debug
-> folder evaluation and human review
-> dataset probe/edit/extract/convert
-> light fine-tune when justified
-> before/after evaluation comparison
```

JSON output is for AI consumption. It does not need to be pleasant for humans. When a command supports `--json`, use it for machine-readable work.

## Operating Assumptions

- Run commands from the repository root.
- Use the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

- Do not assume CUDA. The current baseline is CPU-compatible.
- If GPU verification becomes necessary, decide a separate PyTorch CUDA wheel policy first.
- Heavy or generated paths are intentionally ignored by git:
  - `.venv/`
  - `runs/`
  - `models/`
  - `samples/`
  - `datasets/`
  - `dual-model-operation-kit/`

## Important Local Paths

Typical paths used by the workflow:

```text
models/standard/yolov8s.pt
runs/<eval_id>/
runs/<train_project>/<run_name>/
samples/
datasets/
```

The repository has used tiny smoke artifacts under `runs/id0025_*` and `runs/id0026_*` for pipeline verification. These are ignored and may not exist in a fresh checkout.

## Command Inventory

### Dataset Download

Script:

```text
ai-dataset-download.py
```

Purpose:

- Download small dataset presets for smoke testing.
- Download selected COCO train2017 images from an existing YOLO labels directory.
- Keep full COCO guarded behind `--allow-large` because it is very large. COCO test2017 is additionally gated by `--include-test`.

Examples:

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-download.py --preset coco128-seg --out .\datasets\coco128-seg --json
.\.venv\Scripts\python.exe .\ai-dataset-download.py --preset coco-train-subset --labels .\datasets\coco\labels\train2017 --out .\datasets\coco_subset --classes bench,backpack,handbag,suitcase,refrigerator --max-images-per-class 150 --json
```

Supported presets:

```text
sample-images
coco8
coco8-seg
coco128
coco128-seg
coco
coco-train-subset
```

### Environment Probe

Script:

```text
ai-env-probe.py
```

Purpose:

- Discover Python executable and version.
- Confirm whether the process is running inside a venv.
- Inspect OS, CPU count, RAM, NVIDIA availability, CUDA driver hints.
- Inspect PyTorch, ONNX, ONNXRuntime, Ultralytics, and OpenCV availability.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-env-probe.py --json
```

Use this before blaming model or code behavior on YOLO itself.

### Model Probe

Script:

```text
ai-model-probe.py
```

Arguments:

```text
--model MODEL
--json
```

Purpose:

- Check whether a `.pt` model path exists.
- Try loading with Ultralytics.
- Report YOLO task and class map.
- Preserve load errors as structured data.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-model-probe.py --model .\models\standard\yolov8s.pt --json
```

For fine-tuned smoke models, a successful detect model with classes such as `dog` and `cow` proves that the training artifact is structurally usable. It does not prove accuracy.

### Image Prediction Probe

Script:

```text
ai-image-predict-probe.py
```

Arguments:

```text
--model MODEL
--image IMAGE
--out OUT
--conf CONF
--imgsz IMGSZ
--json
```

Purpose:

- Run one-image inference.
- Return classes, confidence values, boxes, warnings, and errors.
- Optionally save an annotated image.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-image-predict-probe.py --model .\models\standard\yolov8s.pt --image .\samples\test.jpg --out .\runs\probe_one --json
```

Use this for questions like: "Why is this object not detected in this image?"

### ONNX Model Probe

Script:

```text
ai-onnx-probe.py
```

Arguments:

```text
--model MODEL
--providers PROVIDERS
--json
```

Purpose:

- Inspect ONNX model existence, loadability, input/output metadata, and provider compatibility.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-onnx-probe.py --model .\models\exported\model.onnx --providers CPUExecutionProvider --json
```

### ONNX Raw Inference Probe

Script:

```text
ai-onnx-raw-inference-probe.py
```

Arguments:

```text
--model MODEL
--image IMAGE
--providers PROVIDERS
--imgsz IMGSZ
--save-raw
--out OUT
--json
```

Purpose:

- Run raw ONNXRuntime inference.
- Report raw tensor shapes and provider behavior.
- Optionally save raw `.npy` outputs for later analysis.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-onnx-raw-inference-probe.py --model .\models\exported\model.onnx --image .\samples\test.jpg --providers CPUExecutionProvider --save-raw --out .\runs\onnx_raw --json
```

### PyTorch vs ONNX Comparison Probe

Script:

```text
ai-compare-probe.py
```

Arguments:

```text
--pt PT
--onnx ONNX
--image IMAGE
--out OUT
--conf CONF
--imgsz IMGSZ
--providers PROVIDERS
--json
```

Purpose:

- Compare a PyTorch YOLO model and an ONNX model on the same image.
- Produce structured material for diagnosing export/runtime differences.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-compare-probe.py --pt .\models\standard\yolov8s.pt --onnx .\models\exported\model.onnx --image .\samples\test.jpg --providers CPUExecutionProvider --json
```

## Evaluation and Review CLI

Script:

```text
ai-eval.py
```

Subcommands:

```text
evaluate
show
mark
mark-image
summary
export-candidates
```

### Evaluate a Folder

Arguments:

```text
evaluate --model MODEL --input INPUT --out OUT --conf CONF --imgsz IMGSZ --geometry auto|bbox|polygon|obb --overlay auto|bbox|polygon|mask|obb|both --json
```

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py evaluate --model .\models\standard\yolov8s.pt --input .\datasets\images --out .\runs\id0001 --json
```

Output directory shape:

```text
runs/id0001/
  manifest.json
  results.csv
  review.jsonl
  images/
  overlays/
  predictions/
```

`results.csv` stays a lightweight bbox-oriented instance index. Rich instance geometry is stored in `predictions/*.json`: segmentation runs can include `polygon_xy` / `polygon_xyn`, and OBB runs can include `obb_xywhr` / `obb_xyxyxyxy`. `--geometry bbox` suppresses these rich geometry fields while preserving bbox output.

Current geometry support is instance-level: detect, segment, and OBB. Classification and semantic segmentation are future output families because they are image-level or otherwise different from instance geometry, and they are not implemented in `ai-eval.py` yet.

Detection IDs are global within an evaluation run. A human can review an overlay and say:

```text
id0001 det_id 505 is OK
id0001 det_id 506 is false positive
```

### Show One Detection

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py show id0001 505 --json
```

Use this to map a detection ID back to source image, class, confidence, bbox, overlay, and prediction JSON.

### Mark Detection-Level Review

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py mark id0001 506 --status false_positive --note "wrong object" --json
```

Useful statuses:

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

### Mark Image-Level Review

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py mark-image id0001 000238 --status missed --target-class suitcase --note "not detected" --json
```

Use this when the issue is a missed object, not a bad detection row.

### Summarize Reviews

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py summary id0001 --json
```

### Export Review Candidates

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py export-candidates id0001 --status missed,wrong_class,false_positive,bad_box,bad_mask --out .\runs\id0001_candidates --json
```

Use exported candidates as material for dataset correction, conversion, or light fine-tune preparation.

## Validation Probe

Script:

```text
ai-val-probe.py
```

Arguments:

```text
--model MODEL
--data DATA
--out OUT
--imgsz IMGSZ
--conf CONF
--split SPLIT
--json
```

Purpose:

- Run Ultralytics validation and return structured metrics and artifacts.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-val-probe.py --model .\models\standard\yolov8s.pt --data coco128.yaml --out .\runs\val_base --split val --json
```

## Dataset Operations

### Dataset Probe

Script:

```text
ai-dataset-probe.py
```

Arguments:

```text
--path PATH
--json
```

Purpose:

- Detect dataset type.
- Probe YOLO dataset structure and counts.
- Probe conversion artifacts such as COCO JSON or Label Studio exports when supported.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-probe.py --path .\runs\yolo_ready --json
```

### Dataset Convert

Script:

```text
ai-dataset-convert.py
```

Arguments:

```text
--from FMT_FROM
--to FMT_TO
--dataset DATASET
--out OUT
--image-width IMAGE_WIDTH
--image-height IMAGE_HEIGHT
--remap
--images-root IMAGES_ROOT
--classes CLASSES
--force
--json
```

Supported workflow intent:

- YOLO to COCO
- COCO to YOLO
- Label Studio to YOLO
- YOLO to Label Studio

Examples:

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-convert.py --from yolo --to coco --dataset .\runs\yolo_ready --out .\runs\coco_out\annotations.json --json
```

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-convert.py --from coco --to yolo --dataset .\runs\coco_out\annotations.json --out .\runs\yolo_from_coco --remap --json
```

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-convert.py --from labelstudio --to yolo --dataset .\exports\label_studio_tasks.json --out .\runs\yolo_from_ls --classes dog,cow --json
```

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-convert.py --from yolo --to labelstudio --dataset .\runs\yolo_ready --out .\runs\labelstudio_tasks.json --json
```

Prefer `--remap` when preparing a fine-tune dataset from COCO class IDs, because YOLO training generally expects contiguous class IDs starting at 0.

### Dataset Class Edit

Script:

```text
ai-dataset-class-edit.py
```

Arguments:

```text
--dataset DATASET
--out OUT
--from-class FROM_CLASS
--to-class TO_CLASS
--images IMAGES
--labels LABELS
--force
--json
```

Purpose:

- Copy a YOLO dataset and rewrite class IDs in selected labels.
- Useful for simple class correction without opening an annotation GUI.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-class-edit.py --dataset .\runs\yolo_source --out .\runs\yolo_fixed --from-class 1 --to-class 0 --images 000002.jpg --json
```

### Dataset Class Extraction

Script:

```text
ai-dataset-extract-classes.py --dataset DATASET --out OUT --classes CLASSES [--keep-empty-images] [--no-remap] [--stats-only] [--max-images N] [--max-images-per-class N] [--seed N] [--sample-strategy first|random] [--force] [--json]
```

Use `--stats-only` to preview dataset class distribution and bounds before copying large files.
Use `--max-images N` to bound the total number of images.
Use `--max-images-per-class N` to balance a dataset based on per-class image representation limits. Note that extraction distinguishes between 'per-class image count' (images containing the class) and 'annotation occurrence count' (total bounding boxes).
The `--path` argument can be used as an interchangeable alias for `--dataset`.

Purpose:

- Extract only selected classes from a YOLO dataset.
- Optionally remap selected classes to contiguous IDs.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-extract-classes.py --dataset .\runs\yolo_source --out .\runs\yolo_dog_cow --classes dog,cow --json
```

## Light Fine-Tune

Script:

```text
ai-finetune.py
```

Arguments:

```text
--model MODEL
--data DATA
--project PROJECT
--name NAME
--epochs EPOCHS
--imgsz IMGSZ
--batch BATCH
--device DEVICE
--workers WORKERS
--patience PATIENCE
--train-scope full|head
--freeze-layers N
--lr0 LR0
--lrf LRF
--optimizer OPTIMIZER
--cos-lr
--warmup-epochs WARMUP_EPOCHS
--force
--json
```

Purpose:

- Wrap a local Ultralytics YOLO training run.
- Return generated artifact paths as structured JSON.
- Keep noisy Ultralytics logs out of JSON stdout when `--json` is used.
- Allow early stopping patience control for small smoke or subset runs.
- By default, train all layers. Use `--train-scope head` to freeze all layers before the final YOLO head.
- Use `--freeze-layers N` for explicit Ultralytics layer freezing. This overrides `--train-scope head`.
- Use `--lr0`, `--lrf`, `--optimizer`, `--cos-lr`, and `--warmup-epochs` for basic fine-tuning schedule control.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-finetune.py --model .\models\standard\yolov8s.pt --data .\runs\yolo_ready\data.yaml --project .\runs\train --name smoke --epochs 1 --imgsz 64 --batch 1 --device cpu --workers 0 --patience 10 --json
```

Head-only fine-tune example:

```powershell
.\.venv\Scripts\python.exe .\ai-finetune.py --model .\models\standard\yolov8s.pt --data .\runs\yolo_ready\data.yaml --project .\runs\train --name head_only --epochs 5 --imgsz 640 --batch 8 --device cpu --workers 0 --train-scope head --lr0 0.001 --json
```

Important:

- A one-epoch smoke run proves the pipeline, not accuracy.
- Do not claim model improvement unless real validation or evaluation evidence supports it.
- `--patience` controls Ultralytics early stopping. Smaller values can shorten tiny subset experiments, but may stop real training too early when validation metrics are noisy.
- `--train-scope head` is useful for quick adaptation when the existing YOLO backbone is likely good enough.
- Full-layer fine-tuning is still needed when the visual domain is very different from the source model or when head-only training underfits.

## Before/After Evaluation Comparison

Script:

```text
ai-eval-compare-runs.py
```

Arguments:

```text
--before BEFORE
--after AFTER
--json
```

Purpose:

- Compare two `ai-eval.py evaluate` run directories.
- Read `manifest.json`, `results.csv`, and optional `review.jsonl`.
- Return model paths, image counts, detection counts, class counts, confidence summaries, review counts, warnings, and errors.

Example:

```powershell
.\.venv\Scripts\python.exe .\ai-eval-compare-runs.py --before .\runs\eval_base --after .\runs\eval_tuned --json
```

Use this after fine-tuning:

```text
base model evaluation
-> fine-tuned model evaluation
-> compare both run directories
```

## Recommended AI Decision Flow

### User asks: "This object is not detected in this image"

1. Run `ai-env-probe.py --json` if environment is unknown.
2. Run `ai-model-probe.py --model <model> --json`.
3. Run `ai-image-predict-probe.py --model <model> --image <image> --json`.
4. If the image issue is reproducible across a folder, run `ai-eval.py evaluate`.
5. Ask the human to review overlays and mark detections or missed objects.
6. Use review summaries and exported candidates before suggesting dataset edits.
7. Suggest standard-model alternatives or threshold/image-size changes before fine-tune when appropriate.

### User asks: "What training parameters should I use?"

1. Do not jump to hyperparameters.
2. Probe environment and GPU availability.
3. Probe model and dataset.
4. Confirm whether the standard model already solves the problem.
5. If fine-tune is justified, start with a small explicit run and compare before/after evaluations.
6. Treat smoke results as pipeline proof only.

### User asks: "Convert this dataset"

1. Probe the dataset with `ai-dataset-probe.py --path`.
2. Choose the narrow conversion path.
3. Convert with `ai-dataset-convert.py`.
4. Probe the output.
5. If the output is for training, prefer contiguous class IDs.

## What to Share With an AI Assistant

When asking for help, attach or paste:

- `ai-env-probe.py --json` output.
- `ai-model-probe.py --json` output for the relevant model.
- `ai-image-predict-probe.py --json` output for a failing image.
- `ai-eval.py evaluate --json` output and the run path.
- `results.csv`, `manifest.json`, and relevant `predictions/*.json`.
- `ai-dataset-probe.py --path <dataset> --json` output.
- `ai-eval-compare-runs.py --json` output after fine-tune.

## Non-Goals

- This is not a GUI application.
- This is not a fully automated annotation platform.
- This is not a hyperparameter optimization system.
- This is not a guarantee of model accuracy.
- This is not a replacement for human visual review.

## Commit Hygiene

Source and documentation should be committed. Generated artifacts should usually stay ignored.

Before committing, check:

```powershell
git status --short
```

Expected source/document changes may include:

```text
README.md
README_AI.md
docs/AI_WORKFLOW.md
docs/AI_EVAL_REVIEW_CLI_DESIGN.md
docs/AI_DATASET_OPERATIONS_DESIGN.md
ai-*.py
```

Do not stage ignored run/model/sample/dataset artifacts unless explicitly requested.

## License

This repository is released under the MIT License. See `LICENSE`.
