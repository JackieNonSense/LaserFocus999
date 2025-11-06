import cv2
import gc
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from classical_sift.src.detect.selective_search import selective_search_regions
from classical_sift.src.data.prepare_dataset import list_images
from classical_sift.src.data.feature_extract import sift
from classical_sift.configs.config import CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = PROJECT_ROOT / "classical_sift" / "src" / "vocab" / "vocab_k200.pkl"
MODEL_PATH = PROJECT_ROOT / "classical_sift" / "src" / "models" / "svm_bow_model.pkl"

def extract_bow(patch, kmeans):
    kp, desc = sift.detectAndCompute(patch, None)
    if desc is None:
        return np.zeros(kmeans.n_clusters)
    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters+1))
    return hist / np.sum(hist)

def detect_one(img, kmeans, svm, max_regions=200):
    rects = selective_search_regions(img, mode="fast")
    rects = rects[:max_regions]

    best_score = -1
    best_cls = None

    for (x,y,w,h) in rects[:200]:
        patch = img[y:y+h, x:x+w]
        feat = extract_bow(patch, kmeans)
        prob = svm.predict_proba([feat])[0]
        cls_idx = np.argmax(prob)
        score = prob[cls_idx]

        if score > best_score:
            best_score = score
            best_cls = cls_idx

        del patch
        gc.collect()

    return best_cls

if __name__ == "__main__":
    with open(VOCAB_PATH, "rb") as f:
        kmeans = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        svm = pickle.load(f)

    valid_data = list_images("valid")

    total = 0
    correct = 0

    for cls, img_path in tqdm(valid_data[:300]):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (600, int(img.shape[0]*600/img.shape[1])))
        pred = detect_one(img, kmeans, svm)

        if CLASSES[pred] == cls:
            correct += 1
        total += 1

        # must free memory here, otherwise out of memory
        # selective search extremely memory heavy - manual GC is required to avoid OOM on Windows

        del img
        gc.collect()
        cv2.destroyAllWindows()

    print("Selective Search detection accuracy:", correct / total)
