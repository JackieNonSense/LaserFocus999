# Faster R-CNN for AgroPest-12 Insect Detection

Faster R-CNN implementation using Detectron2 for insect detection and classification.

## Directory Structure

```
faster-rcnn/
├── configs/                    # Configuration files
│   └── faster_rcnn_R50_FPN.yaml
├── src/                        # Source code
│   ├── data/                   # Dataset handling
│   │   └── dataset.py         # Dataset registration
│   ├── models/                 # Model definitions
│   ├── engine/                 # Training engine
│   └── utils/                  # Utility functions
├── scripts/                    # Executable scripts
│   ├── yolo_to_coco.py        # Format conversion
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── notebooks/                  # Jupyter notebooks
├── outputs/                    # Training outputs (not in git)
│   ├── checkpoints/           # Model checkpoints
│   ├── logs/                  # Training logs
│   ├── results/               # Evaluation results
│   └── coco_annotations/      # Converted COCO format data
├── requirements.txt           # Python dependencies
└── environment.yml            # Conda environment
```

## Setup

### 1. Environment Setup

**Option A: Using conda**
```bash
conda env create -f environment.yml
conda activate insect-detection
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

### 2. Install PyTorch

Install PyTorch according to your CUDA version:
```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check: https://pytorch.org/get-started/locally/

### 3. Install Detectron2

**From source (recommended):**
```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

**Or via pip:**
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

See: https://detectron2.readthedocs.io/en/latest/tutorials/install.html

## Usage

### Step 1: Convert YOLO to COCO Format

The AgroPest-12 dataset uses YOLO format. Convert it to COCO format for Detectron2:

```bash
python scripts/yolo_to_coco.py \
    --yolo_dir ../data/AgroPest-12 \
    --output_dir outputs/coco_annotations \
    --splits train valid test
```

This creates:
- `outputs/coco_annotations/train_coco.json`
- `outputs/coco_annotations/valid_coco.json`
- `outputs/coco_annotations/test_coco.json`

### Step 2: Train Model

```bash
python scripts/train.py \
    --config-file configs/faster_rcnn_R50_FPN.yaml \
    --data-root ../data/AgroPest-12 \
    --coco-json-dir outputs/coco_annotations \
    --num-gpus 1
```

**Monitor training:**
```bash
tensorboard --logdir outputs/checkpoints/faster_rcnn_R50_FPN
```

### Step 3: Evaluate Model

```bash
python scripts/evaluate.py \
    --config-file configs/faster_rcnn_R50_FPN.yaml \
    --weights outputs/checkpoints/faster_rcnn_R50_FPN/model_final.pth \
    --data-root ../data/AgroPest-12 \
    --coco-json-dir outputs/coco_annotations \
    --split test
```

## Configuration

Edit `configs/faster_rcnn_R50_FPN.yaml` to adjust:

- **Batch size**: `SOLVER.IMS_PER_BATCH` (reduce if GPU memory insufficient)
- **Learning rate**: `SOLVER.BASE_LR`
- **Training iterations**: `SOLVER.MAX_ITER`
- **Image size**: `INPUT.MIN_SIZE_TRAIN`, `INPUT.MAX_SIZE_TRAIN`

## Expected Results

Training will produce:
- Model checkpoints in `outputs/checkpoints/`
- Training logs in `outputs/logs/`
- Evaluation results in `outputs/results/`

Evaluation metrics:
- mAP (COCO-style)
- AP@0.5, AP@0.75
- Per-class AP

## Troubleshooting

**CUDA out of memory:**
- Reduce `SOLVER.IMS_PER_BATCH` in config
- Reduce `INPUT.MIN_SIZE_TRAIN`

**Dataset not found:**
- Ensure COCO format conversion completed successfully
- Check paths in `--data-root` and `--coco-json-dir`

**Import errors:**
- Ensure Detectron2 is installed correctly
- Run `python -c "import detectron2; print(detectron2.__version__)"`

## Next Steps

1. Convert dataset format
2. Start training
3. Monitor with TensorBoard
4. Evaluate on test set
5. Compare with YOLO results in `../comparison/`
