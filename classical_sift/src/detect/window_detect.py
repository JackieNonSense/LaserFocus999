import cv2
import numpy as np
import pickle
from pathlib import Path

from classical_sift.src.data.feature_extract import sift
from classical_sift.src.detect.window_generator import generate_windows

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = PROJECT_ROOT / "classical_sift" / "src" / "vocab" / "vocab_k200.pkl"
MODEL_PATH = PROJECT_ROOT / "classical_sift" / "src" / "models" / "svm_bow_model.pkl"


def extract_bow_from_patch(patch, kmeans):
    kp, desc = sift.detectAndCompute(patch, None)
    if desc is None:
        return np.zeros(kmeans.n_clusters)
    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters+1))
    return hist / np.sum(hist)


def window_detect(img, kmeans, svm, win_ratio=0.25, stride_ratio=0.125):
    windows = generate_windows(img, win_ratio, stride_ratio)

    best_score = -1
    best_cls = None
    best_box = None

    for (x1,y1,x2,y2) in windows:
        patch = img[y1:y2, x1:x2]
        feat = extract_bow_from_patch(patch, kmeans)
        
        prob = svm.predict_proba([feat])[0]   # e.g. [0.1,0.3,0.6]
        cls_idx = np.argmax(prob)
        score = prob[cls_idx]

        if score > best_score:
            best_score = score
            best_cls = cls_idx
            best_box = (x1,y1,x2,y2)

    return best_cls, best_score, best_box
