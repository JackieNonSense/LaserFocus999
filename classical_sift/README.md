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

### 1. Build Visual Vocabulary

(based on SIFT descriptors)

```bash
python -m classical_sift.src.data.build_vocab
```

### 2. Train BOW + SVM Classifier

```bash
python -m classical_sift.src.models.svm_bow
```

### 3. Detection using Selective Search (Optional Extension)

This method also implements a Selective Search based detection pipeline as an experimental extension.
**Warning**: On some Windows setups, OpenCV Selective Search may cause memory issues during large-batch evaluation. As per the spec’s allowance to use Colab, the full Selective Search evaluation results (detection accuracy) were generated on Colab Linux runtime.
This is a known limitation of classical CV pipelines (pre-deep-learning era).
For reproducible baseline, we keep sliding-window detection runnable locally.

```bash
python -m classical_sift.scripts.eval_ss
```

This will output detection accuracy on the validation set.

## Notes

- Models (`*.pkl`) are **not included in git** (per project spec)
- Dataset path is expected at: `data/AgroPest-12`
- This method runs slower than deep models (Selective Search is expensive) but is useful as an interpretable classical baseline

## Next Steps

After generating predictions, final evaluation can be run using group shared script:

```
comparison/scripts/unified_evaluation.py
```
