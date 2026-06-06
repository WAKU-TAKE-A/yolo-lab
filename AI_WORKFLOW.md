# AI-First YOLO Workflow

本ドキュメントは、YOLOモデルの操作、データセット変換、ファインチューニング、および評価をAIアシスタントが自律的かつ確実に行うためのワークフローとコマンドセットを説明します。

## 概要 (Overview)
このワークフローは、GUIや汎用的な学習プラットフォームを提供することを目的としていません。代わりに、AIが安定して操作し、結果をJSON形式で構造化して受け取ることができるコマンドラインツール群を提供します。これにより、ONNXRuntime連携や追加学習に関するAIとの対話において、「ハルシネーション（幻覚）」を防ぎ、確実な事実ベースの作業が可能になります。

## ワークフローの6つのステージ (The Big Flow)

### 1. PC/Environment/Model Check (環境とモデルの診断)
システムの前提条件（Python, PyTorch, Ultralytics, CUDAの有無）や、対象のYOLOモデル（クラス構成、タスク）を調査します。
- `ai-env-probe.py`
- `ai-model-probe.py`
- `ai-onnx-probe.py`
- `ai-onnx-raw-inference-probe.py`
**例:**
```powershell
.\.venv\Scripts\python.exe .\ai-model-probe.py --model .\models\standard\yolov8s.pt --json
```

### 2. Image-level Prediction/Debug (画像レベルの推論・デバッグ)
1枚の画像に対して推論を行い、結果（座標、信頼度、クラス）を抽出します。
- `ai-image-predict-probe.py`
**例:**
```powershell
.\.venv\Scripts\python.exe .\ai-image-predict-probe.py --model .\models\standard\yolov8s.pt --image .\samples\test.jpg --json
```

### 3. Evaluation and Human Review (評価とヒューマンレビュー)
フォルダ内の全画像を評価し、推論結果を保存します。オプションで、人間が正誤を判定するためのレビューファイルを生成します。
- `ai-eval.py`
- `ai-val-probe.py`
- `ai-compare-probe.py`
**例:**
```powershell
.\.venv\Scripts\python.exe .\ai-eval.py evaluate --model .\models\standard\yolov8s.pt --input .\datasets\images --out .\runs\eval_base --json
```

### 4. Dataset Probing/Edit/Extraction/Conversion (データセットの調査・編集・抽出・変換)
YOLO、COCO、Label Studio間のデータ形式の変換や、クラスIDの編集、特定クラスの抽出による学習用データセットの構築を行います。
- `ai-dataset-probe.py`
- `ai-dataset-convert.py`
- `ai-dataset-class-edit.py`
- `ai-dataset-extract-classes.py`
**例:**
```powershell
.\.venv\Scripts\python.exe .\ai-dataset-extract-classes.py --dataset .\runs\yolo_source --out .\runs\yolo_ready --classes dog,cow --json
```

### 5. Light Fine-Tune (軽量ファインチューニング)
抽出・整形されたYOLOデータセットを用いて、対象モデルに追加学習（ファインチューニング）を適用します。
- `ai-finetune.py`
**例:**
```powershell
.\.venv\Scripts\python.exe .\ai-finetune.py --model .\models\standard\yolov8s.pt --data .\runs\yolo_ready\data.yaml --project .\runs\train --name run1 --json
```

### 6. Before/After Evaluation Comparison (学習前後の評価比較)
ベースモデルとファインチューニング後のモデルの評価結果を比較し、変化をJSONで出力します。
- `ai-eval-compare-runs.py`
**例:**
```powershell
.\.venv\Scripts\python.exe .\ai-eval-compare-runs.py --before .\runs\eval_base --after .\runs\eval_tuned --json
```

## 重要な原則と制約 (Principles & Limitations)
1. **JSON出力はAI用です:** 全てのコマンドで`--json`フラグを使用すると、人間にとっての読みやすさよりも、AI（LLM）が確実にパースできる構造化データが出力されます。
2. **Smoke Fine-Tuneはパイプラインの検証用です:** `ai-finetune.py`による軽量学習は、ワークフロー全体が繋がっているかを検証（スモークテスト）するためのものであり、高精度なモデルを保証するものではありません。
3. **GPU (CUDA) サポート:** 本環境はCPUベースで構築されています。将来的にGPUを用いた本格的な検証を行う場合は、別途PyTorchのCUDA Wheel（GPU対応パッケージ）のインストールポリシーを定める必要があります。
