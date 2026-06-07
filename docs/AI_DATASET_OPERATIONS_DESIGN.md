# AI Dataset Operations Design

Last updated: 2026-06-06

## Purpose

The purpose of dataset operations is to provide generative AI agents with reliable facts and safe programmatic handles for dataset inspection, class label corrections, category-based extraction, and format conversions *before* triggering any fine-tuning.

This makes the dataset preparation and correction workflow reproducible and addressable via simple CLI commands, rather than relying on complex manual GUI interaction or ad-hoc custom scripts.

## Non-Goals

- **No GUI**: Do not build a graphical annotation or dataset visualization interface.
- **No full annotation suite**: This is not a replacement for full labeling tools (e.g. CVAT, Label Studio) for geometry creation.
- **No training platform**: We are not building a generic model training pipeline. Training parameters and optimization are out of scope here.

## Evaluation Handoff (`ai-eval export-candidates`)

`ai-eval export-candidates` generates candidates under the run directory containing:
- `candidates.json`
- `candidates.csv`
- `images.txt`

These files list exact failure items (such as `missed` detections, `false_positive` bboxes, `wrong_class` labels, etc.) decorated with image, class, and prediction data. Later dataset operation commands (like `ai-dataset-class-edit`) will accept these handoff artifacts directly to apply targeted label fixes without altering bounding geometries.

## Planned Command Flow

The dataset operation workflow is organized into sequential commands:

### 1. `ai-dataset-probe` (Current Phase: Read-Only)
Inspects a directory or a file, detects its format, and reports layout statistics, categories, image counts, label counts, and candidate facts. 

### 2. `ai-dataset-class-edit` (Future)
Performs class label changes on detection/segmentation annotations (e.g., changing class Y to class X for selected images or global mappings) while preserving the underlying coordinate geometries.

### 3. `ai-dataset-extract-classes` (Future)
Creates a focused subset of an existing dataset (such as YOLO or COCO formats) by extracting only specified classes/images, discarding unrelated categories to optimize downstream fine-tuning.

### 4. `ai-dataset-convert` (Future)
Converts annotations between formats (YOLO format `.txt`, COCO JSON, and Label Studio exports) to bridge tool compatibility gaps.

### 5. `ai-finetune` (Future)
Runs light additional training on the standard model using the clean, filtered, and corrected datasets generated in the previous steps.

---

> [!IMPORTANT]
> **Read-Only Status**: The current implementation phase (`ai-dataset-probe`) is strictly read-only. It inspects and reports dataset layouts, category sets, and file metrics without modifying any datasets or files.
