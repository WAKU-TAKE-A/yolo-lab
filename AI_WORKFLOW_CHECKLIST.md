# AI Workflow Checklist

このチェックリストは、AIワークフローを安全かつ確実に進めるための実用的なガイドです。

## 1. 必須ファイルとモデルの確認 (Required Local Files/Models)
- [ ] `models/standard/yolov8s.pt` 等のベースモデルが存在するか？
- [ ] 仮想環境(`.venv`)が有効化されており、`ultralytics` がインストールされているか？
- [ ] 各コマンドスクリプト（`ai-*.py`）がプロジェクト直下に存在するか？

## 2. スタンダードモデル利用の判断 (Standard-Model-First Decision Point)
- 新しい物体検出タスクを始める際は、**常に**既存の標準モデル（例: `yolov8s.pt`）で `ai-eval.py` を実行し、既存の推論能力を確認すること。
- いきなり追加学習（ファインチューニング）を始めない。

## 3. データセット操作への移行タイミング (When to Move to Dataset Operation)
- 標準モデルでの評価後、特定クラスの検出漏れや誤検知が多い場合にデータセット操作（変換・編集・抽出）へ移行する。
- 例: Label Studioで作成したアノテーションをYOLO形式に変換（`ai-dataset-convert.py`）し、目的のクラスだけを抽出（`ai-dataset-extract-classes.py`）して学習セットを構築する。

## 4. ファインチューニングの実行タイミング (When Light Fine-Tune is Reasonable)
- 学習用データセット（YOLO形式）が正しく構築され、`ai-dataset-probe.py` によってエラーなく認識された場合。
- 目的が「パイプラインの検証（Smoke Test）」または「少数の追加クラスの概念実証（MVP）」である場合。本格的な精度向上には、GPU環境での長時間の学習が必要です。

## 5. AIへの共有事項 (What to Attach/Share with AI)
AIアシスタントにエラーや現状の相談をする際は、以下のJSONアーティファクトを共有してください：
- **環境情報:** `ai-env-probe.py --json` の出力結果
- **モデル情報:** `ai-model-probe.py --model <対象モデル> --json` の出力結果
- **データセット状態:** `ai-dataset-probe.py --path <対象パス> --json` の出力結果
- **評価比較:** `ai-eval-compare-runs.py --json` の出力結果
- 発生した例外エラーや、意図しないJSONの構造
