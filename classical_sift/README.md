# Classical SIFT + Bag-of-Words + SVM Baseline for AgroPest-12

Classical CV baseline using SIFT descriptors + Visual Vocabulary (Bag-of-Words) + SVM classifier, and Selective Search for region proposals.

This method provides a non-deep-learning contrast baseline to compare against YOLO and Faster R-CNN.

## Directory Structure

```
classical_sift/
├── configs/                   # Config (class list, modes)
├── src/
│   ├── data/                  # Dataset loading / features
│   ├── vocab/                 # Vocabulary building
│   ├── models/                # SVM classifier
│   ├── detect/                # Detection (Selective Search)
│   └── eval/                  # Evaluation scripts
```

## Setup

```bash
pip install -r requirements.txt
```

SIFT requires `opencv-contrib-python` (not included in default OpenCV package):

```bash
pip install opencv-contrib-python
```

## Usage

**Pipeline**

1. SIFT feature extraction
2. Build visual vocabulary using MiniBatchKMeans (Bag-of-Words)
3. Train RBF SVM classifier on BoW histograms
4. Use Selective Search as detector + SVM classification for detected regions

This satisfies the requirement of "Detector + Classifier" forming one complete method.

### Step 1: Build Visual Vocabulary

(based on SIFT descriptors)

```bash
python -m classical_sift.src.data.build_vocab
```

### Step 2: Train BOW + SVM Classifier

```bash
python -m classical_sift.src.models.svm_bow
```

### Step 3: Detection using Selective Search (Optional Extension)

This method also implements a Selective Search based detection pipeline as an experimental extension.  
**Warning**: On some Windows setups, OpenCV Selective Search may cause memory issues during large-batch evaluation. As per the spec’s allowance to use Colab, the full Selective Search evaluation results (detection accuracy) were generated on Colab Linux runtime.
This is a known limitation of classical CV pipelines (pre-deep-learning era).
For reproducible baseline, we keep sliding-window detection runnable locally.

```bash
python -m classical_sift.scripts.eval_ss
```

This will output detection accuracy and result json on the validation set.

### Step 4: Evaluate Model

Linux:

```bash
python comparison/scripts/unified_evaluation.py \
  --method classical-sift \
  --predictions classical_sift/results/predictions_ss_valid200.json \
  --ground_truths classical_sift/results/gt_valid200_simple.json \
  --class_names Ants Bees Beetles Caterpillars Earthworms Earwigs Grasshoppers Moths Slugs Snails Wasps Weevils

```

or Windows:

```bash
python comparison\scripts\unified_evaluation.py --method classical-sift --predictions classical_sift/results/predictions_ss_valid200.json --ground_truths classical_sift/results/gt_valid200_simple.json --class_names Ants Bees Beetles Caterpillars Earthworms Earwigs Grasshoppers Moths Slugs Snails Wasps Weevils

```

## Notes

- Models (`*.pkl`) are **not included in git** (per project spec)
- Dataset path is expected at: `data/AgroPest-12`
- This method runs slower than deep models (Selective Search is expensive) but is useful as an interpretable classical baseline
- Weak performance on fine-grained insect classes (expected in literature)
- This baseline is mainly used for comparison purpose against deep-learning models

## Evaluation

We provide standard outputs compatible with `/comparison/scripts/unified_evaluation.py`:

- predictions JSON
- simplified ground truth JSON

Selective Search detection evaluation is runnable but may require Colab runtime due to memory limit on Windows PC.

Example result (valid set 12-classes):
Accuracy: ~0.55
