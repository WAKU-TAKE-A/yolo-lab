# YOLO-Lab

YOLO-Lab is an AI-friendly command-line toolkit for YOLO model inspection, ONNXRuntime checks, evaluation review, dataset operations, and lightweight fine-tuning.

YOLO-Lab は、YOLO / ONNXRuntime / 評価 / データセット変換 / 軽量 fine-tune を、生成AIが扱いやすい形で進めるためのコマンド集です。

目的は GUI や高機能な統合アプリを作ることではありません。生成AIが環境やモデルを誤解せず、JSON などの構造化出力を材料にして、より安定した回答やスクリプト提案を返せるようにすることです。

AI が読む詳細仕様は [README_AI.md](README_AI.md) にあります。

## 基本方針

まず標準モデルで確認します。いきなり追加学習に進みません。

大きな流れは次の通りです。

```text
1. PC環境・モデル確認
2. 画像単位の推論確認
3. フォルダ評価と人間レビュー
4. データセット調査・編集・抽出・変換
5. 軽量 fine-tune
6. 学習前後の評価比較
```

`--json` 出力は主に AI が読むためのものです。人間に見やすいことより、AI がパースしやすいことを優先しています。

## セットアップ

このリポジトリでは venv を使います。

```powershell
.\setup_venv.bat
```

通常の実行は次の Python を使います。

```powershell
.\.venv\Scripts\python.exe
```

GPU 前提の検証に進む場合は、PyTorch CUDA wheel の扱いを別途決める必要があります。現状の基本は CPU でも動く確認フローです。

## よく使うコマンド

環境確認:

```powershell
.\.venv\Scripts\python.exe .\ai-env-probe.py --json
```

データセット取得:

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-download.py --preset coco128-seg --out .\datasets\coco128-seg --json
```

モデル確認:

```powershell
.\.venv\Scripts\python.exe .\ai-model-probe.py --model .\models\standard\yolov8s.pt --json
```

1枚の画像で推論確認:

```powershell
.\.venv\Scripts\python.exe .\ai-image-predict-probe.py --model .\models\standard\yolov8s.pt --image .\samples\test.jpg --out .\runs\probe_one --json
```

画像フォルダを評価:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py evaluate --model .\models\standard\yolov8s.pt --input .\datasets\images --out .\runs\id0001 --json
```

検出IDの詳細確認:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py show id0001 505 --json
```

誤検出などを記録:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py mark id0001 506 --status false_positive --note "誤検出" --json
```

未検出などを画像単位で記録:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py mark-image id0001 000238 --status missed --target-class suitcase --note "右下が未検出" --json
```

未検出の正しい geometry を追加:

```powershell
.\.venv\Scripts\python.exe .\ai-eval.py add-annotation id0001 000238 --class suitcase --geometry bbox --bbox 120,80,300,260 --json
```

`000238` は検出IDではなく画像IDです。検出が0件の画像でも、評価run内に画像IDがあれば `add-annotation` で bbox / polygon / OBB を追加できます。追加された annotation は `review.jsonl` に追記され、`results.csv` と `predictions/*.json` は書き換えません。

データセット確認:

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-probe.py --path .\runs\yolo_ready --json
```

YOLO から COCO へ変換:

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-convert.py --from yolo --to coco --dataset .\runs\yolo_ready --out .\runs\coco_out\annotations.json --json
```

クラスを修正:

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-class-edit.py --dataset .\runs\yolo_source --out .\runs\yolo_fixed --from-class 1 --to-class 0 --images 000002.jpg --json
```

- **Dataset Utilities**: Check dataset integrity, perform bounded subset extraction (`--max-images`, `--max-images-per-class`), remap classes, and convert formats.

特定クラスだけ抽出 (データセットのパスは --dataset または --path で指定可能):

```powershell
.\.venv\Scripts\python.exe .\ai-dataset-extract-classes.py --path .\runs\yolo_source --out .\runs\yolo_dog_cow --classes dog,cow --json
```

軽量 fine-tune:

```powershell
.\.venv\Scripts\python.exe .\ai-finetune.py --model .\models\standard\yolov8s.pt --data .\runs\yolo_ready\data.yaml --project .\runs\train --name smoke --epochs 1 --imgsz 64 --batch 1 --device cpu --workers 0 --patience 10 --json
```

ヘッド層だけを学習:

```powershell
.\.venv\Scripts\python.exe .\ai-finetune.py --model .\models\standard\yolov8s.pt --data .\runs\yolo_ready\data.yaml --project .\runs\train --name head_only --epochs 5 --imgsz 640 --batch 8 --device cpu --workers 0 --train-scope head --lr0 0.001 --json
```

学習前後の評価比較:

```powershell
.\.venv\Scripts\python.exe .\ai-eval-compare-runs.py --before .\runs\eval_base --after .\runs\eval_tuned --json
```

## 評価結果の考え方

`ai-eval.py evaluate` は評価ごとに `runs/<eval_id>/` を作ります。

```text
runs/id0001/
  manifest.json
  results.csv
  review.jsonl
  images/
  overlays/
  predictions/
```

画像ビュアーで `overlays/` を見ながら、検出IDに対してレビューできます。

`results.csv` は bbox ベースの軽量な検出インデックスとして維持します。segmentation の polygon や OBB の回転矩形などの詳細 geometry は `predictions/*.json` に保存します。`--geometry bbox` を指定すると詳細 geometry は保存せず、bbox のみを扱います。

現在の `ai-eval.py` の対象は instance-level の detect / segment / OBB です。classify と semantic segmentation は画像単位または別形式の出力なので、将来対応予定として扱い、現時点では未実装です。

例:

```text
id0001 の 505 は OK
id0001 の 506 は誤検出
id0001 の 000238 は suitcase 未検出
```

このレビュー結果をもとに、データセット修正や fine-tune に進むか判断します。

## fine-tune について

このリポジトリの fine-tune は、まず軽量な追加学習フローを確実に通すためのものです。

デフォルトでは Ultralytics YOLO の通常学習として全層を学習します。ヘッド層のみを学習したい場合は `--train-scope head` を使います。より細かく制御したい場合は `--freeze-layers N` で先頭から N 層を freeze できます。

少量データで追加学習する場合は、まず `--train-scope head --lr0 0.001` から始めることを推奨します。全層学習は既存特徴まで動くため、データが少ないと汎用特徴を崩す可能性があります。head-only で明らかに足りない場合や、元モデルと対象ドメインが大きく異なる場合に、全層学習を次の段階として検討します。

1 epoch の smoke 実行で「精度が上がった」とは判断しません。精度について話す場合は、標準モデル評価、fine-tuned モデル評価、比較結果をセットで見ます。

## 主なファイル

- `README_AI.md`: AI 向けの完全・詳細な英語仕様
- `docs/AI_WORKFLOW.md`: ワークフロー概要と実用チェックリスト
- `ai-env-probe.py`: 環境確認
- `ai-model-probe.py`: モデル確認
- `ai-image-predict-probe.py`: 画像単位の推論確認
- `ai-eval.py`: 評価・レビュー管理
- `ai-dataset-probe.py`: データセット確認
- `ai-dataset-download.py`: sample / COCO 系データセット取得
- `ai-dataset-convert.py`: YOLO / COCO / Label Studio 変換
- `ai-dataset-class-edit.py`: YOLO ラベルのクラス修正
- `ai-dataset-extract-classes.py`: 特定クラス抽出
- `ai-finetune.py`: 軽量 fine-tune
- `ai-eval-compare-runs.py`: 評価結果の before/after 比較
- `ai-onnx-probe.py`: ONNX モデル確認
- `ai-onnx-raw-inference-probe.py`: ONNXRuntime 生推論確認
- `ai-compare-probe.py`: PyTorch YOLO と ONNX の比較材料取得
- `ai-val-probe.py`: Ultralytics validation 確認

## Git 管理

生成物や大きいファイルは基本的にコミットしません。

主に無視されるもの:

```text
.venv/
runs/
models/
samples/
datasets/
dual-model-operation-kit/
```

コミット前は確認します。

```powershell
git status --short
```

## License

MIT License. See [LICENSE](LICENSE).
