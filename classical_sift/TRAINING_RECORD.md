# Classical SIFT + Bag-of-Words + SVM Training Record

## Method Summary

A classical computer vision baseline using:

- **Feature extractor**: SIFT
- **Vocabulary building**: Bag-of-Words using MiniBatchKMeans
- **Classifier**: SVM (RBF kernel)
- **Detector**: Selective Search (fast mode)

This baseline is used as a comparison point against modern deep learning detection pipelines.

## Dataset

| Split | Images |
| ----- | ------ |
| Train | 11,502 |
| Valid | 1,095  |

Dataset: AgroPest-12 (12 insect classes)

To reduce memory load, vocabulary built using sampled subset ≈ ~2,000 SIFT descriptor sources (~ <20% total dataset).

## Training Settings

| Component       | Config                       |
| --------------- | ---------------------------- |
| SIFT            | default OpenCV SIFT_create() |
| Vocabulary size | K = 200                      |
| Classifier      | SVM (RBF), probability=True  |
| Normalization   | L1 normalized BOW histogram  |
| Detection       | Selective Search fast        |

Hardware (Local Windows Desktop)

- CPU only
- SIFT + SS extremely memory intensive, manual GC required
- Full dataset selective search not runnable locally → validated on subset (400 images)

Later moved evaluation to Google Colab for stable runtime environment (colab recommended computational environment).

## Training & Validation Results

| Stage                                                           | Accuracy   |
| --------------------------------------------------------------- | ---------- |
| Classification only (12-class)                                  | **0.3908** |
| Selective Search Detection evaluation (subset 400 valid images) | **0.55**   |

Note: classical CV is not expected to match deep learning performance. Result shown here is consistent with known literature where BoW+SVM underperforms significantly on fine-grained insect classification.

## Limitations Observed

- Selective search causes memory spikes → often OOM on Windows
- Descriptor distribution extremely sparse per region → many patches uninformative
- Very sensitive to intra-class similarity (AgroPest-12 is highly similar fine-grained species)
- Modern datasets significantly favor deep-learning based end-to-end feature learning

## Conclusion

This method serves as a historical / classical baseline reference point.
It demonstrates how manually engineered features struggle on complex fine-grained domains, validating the necessity and advantage of modern CNN based detectors in this dataset.
