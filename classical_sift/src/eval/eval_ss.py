import cv2
import gc
import json
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
RESULTS_DIR = PROJECT_ROOT / "classical_sift" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LIMIT = 200       
RESIZE_W = 600  
MAX_REGIONS = 200   

def extract_bow(patch, kmeans):
    # SIFT on gray
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(patch, None)
    if desc is None or len(desc) == 0:
        return np.ones(kmeans.n_clusters, dtype=np.float32) / kmeans.n_clusters
    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
    s = hist.sum()
    if s == 0:
        return np.ones_like(hist, dtype=np.float32) / len(hist)
    return (hist / s).astype(np.float32)

def detect_one(img, kmeans, svm, max_regions=MAX_REGIONS):
    rects = selective_search_regions(img, mode="fast")
    rects = rects[:max_regions]

    best_score = -1.0
    best_cls = None
    best_box = None  # [x1, y1, x2, y2]

    for (x, y, w, h) in rects:
        if w < 20 or h < 20:
            continue
        patch = img[y:y+h, x:x+w]
        feat = extract_bow(patch, kmeans)
        prob = svm.predict_proba([feat])[0] 
        cls_idx = int(np.argmax(prob))
        score = float(prob[cls_idx])
        if score > best_score:
            best_score = score
            best_cls = cls_idx
            best_box = [int(x), int(y), int(x + w), int(y + h)]
        del patch
        gc.collect()

    return best_cls, best_score, best_box

def resize_keep_w(img, target_w=RESIZE_W):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    new_h = int(h * (target_w / float(w)))
    return cv2.resize(img, (target_w, new_h))


if __name__ == "__main__":
    # load models
    with open(VOCAB_PATH, "rb") as f:
        kmeans = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        svm = pickle.load(f)

    valid_data = list_images("valid")

    total = 0
    correct = 0    
    predictions = []  # for unified_evaluation
    gt_simple = []    

    for cls, img_path in tqdm(valid_data[:LIMIT]):
        img = cv2.imread(str(img_path))
        if img is None:
            total += 1
            predictions.append({
                "image_id": Path(img_path).name,
                "boxes": [[0, 0, 1, 1]],
                "labels": [0],
                "scores": [0.0],
            })
            gt_simple.append({
                "image_id": Path(img_path).name,
                "labels": [int(CLASSES.index(cls))]
            })
            continue

        img = resize_keep_w(img, RESIZE_W)
        pred_idx, score, box = detect_one(img, kmeans, svm, max_regions=MAX_REGIONS)

        if pred_idx is not None and CLASSES[pred_idx] == cls:
            correct += 1
        total += 1

        if pred_idx is not None and CLASSES[pred_idx] == cls:
            correct += 1
        total += 1

        # Record unified evaluation format
        if box is None:
            h, w = img.shape[:2]
            box = [0, 0, w - 1, h - 1]
            if pred_idx is None:
                pred_idx = 0
                score = 0.0

        predictions.append({
            "image_id": Path(img_path).name,
            "boxes": [box],
            "labels": [int(pred_idx)],
            "scores": [float(score)],
        })

        gt_simple.append({
            "image_id": Path(img_path).name,
            "labels": [int(CLASSES.index(cls))]
        })


        # must free memory here, otherwise out of memory
        # selective search extremely memory heavy - manual GC is required to avoid OOM on Windows

        # del img
        # gc.collect()
        # cv2.destroyAllWindows()


    acc = correct / max(total, 1)
    print(f"Selective Search detection accuracy on {total} images: {acc:.6f}")

    # Save predictions / gt（for unified evaluation）
    pred_path = RESULTS_DIR / f"predictions_ss_valid{LIMIT}.json"
    gt_path   = RESULTS_DIR / f"gt_valid{LIMIT}_simple.json"
    with open(pred_path, "w") as f:
        json.dump({"predictions": predictions}, f)
    with open(gt_path, "w") as f:
        json.dump(gt_simple, f)
    print("Saved predictions:", pred_path)
    print("Saved ground truths:", gt_path)

