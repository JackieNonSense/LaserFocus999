# Repository Guidelines

## Project Structure & Module Organization
LaserFocus999 hosts multiple baselines under sibling directories. `faster-rcnn/` contains Detectron2 configs, scripts, and `src/` code; `classical_sift/` houses the SIFT→BoW→SVM pipeline; `yolov8s/` stores Ultralytics configs, metrics CSVs, and notebooks; `comparison/` centralizes evaluation scripts/results; `utils/` provides shared helpers like `gradcam.py`. Keep datasets in `data/AgroPest-12/` (untracked) and place generated artifacts inside each method’s `outputs/` or `results/` subfolders to keep the root clean.

## Build, Test, and Development Commands
- `conda env create -f faster-rcnn/environment.yml && conda activate insect-detection` – Provision the Detectron2 training environment.
- `pip install -r classical_sift/requirements.txt` – Install dependencies for the classical baseline (requires OpenCV-contrib).
- `python faster-rcnn/scripts/yolo_to_coco.py --yolo_dir data/AgroPest-12 --output_dir faster-rcnn/outputs/coco_annotations --splits train valid test` – Convert YOLO annotations before Faster R-CNN work.
- `python faster-rcnn/scripts/train.py --config-file faster-rcnn/configs/faster_rcnn_R50_FPN.yaml --data-root data/AgroPest-12 --coco-json-dir faster-rcnn/outputs/coco_annotations` – Launch Faster R-CNN training; append `--resume` to continue.
- `python comparison/scripts/unified_evaluation.py --method <name> --predictions <json> --ground_truths <json> --class_names <list>` – Produce comparable metrics for every method.

## Coding Style & Naming Conventions
Target Python 3.10+, 4-space indentation, and PEP 8 import grouping (stdlib, third-party, local). Prefer snake_case modules/functions, PascalCase classes, and kebab-case experiment folders (e.g., `outputs/checkpoints/faster_rcnn_R50_FPN`). Keep configs lowercase with underscores. Ensure scripts run via `python path/to/script.py` using `argparse` for parameters, and reserve comments/docstrings for non-obvious logic.

## Testing & Evaluation Expectations
No unit-test harness exists; regressions are detected via deterministic evaluations. Before opening a PR, rerun the relevant validation pass (`python faster-rcnn/scripts/evaluate.py --split valid` or `python classical_sift/scripts/eval_ss.py`) and refresh `comparison/results/` artifacts. Record mAP, AP50/75, APs/m/l, and runtime stats inside the appropriate `TRAINING_RECORD.md` entry so peers can reproduce outcomes.

## Commit & Pull Request Guidelines
Use the existing short, imperative commit style (`Add YOLO val summary`, `Update README`). Work on feature branches named `YourName/Feature`. Each PR should include a concise summary, reproduction commands, updated metric tables or CSV/JSON links, and screenshots when visualizations change. Do not commit datasets, checkpoints, or secrets; confirm new large files are ignored before pushing.

## Data & Configuration Hygiene
Store raw data strictly inside `data/AgroPest-12/`. Keep Kaggle tokens, cloud credentials, and other secrets in local `.env` files already ignored by git. When sharing configs, prefer relative paths rooted at the repo top level and strip machine-specific settings to keep setups portable.
