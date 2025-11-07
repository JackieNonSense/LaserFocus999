import cv2
import pickle
import numpy as np
from pathlib import Path

from classical_sift.src.detect.selective_search import selective_search_regions
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


def detect_image_ss(img_path, max_regions=200):
    img = cv2.imread(str(img_path))
    with open(VOCAB_PATH, "rb") as f:
        kmeans = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        svm = pickle.load(f)

    rects = selective_search_regions(img, mode="fast")

    best_score = -1
    best_cls = None
    best_box = None

    # limit region number for speed
    rects = rects[:max_regions]

    for (x,y,w,h) in rects:
        patch = img[y:y+h, x:x+w]
        feat = extract_bow(patch, kmeans)
        prob = svm.predict_proba([feat])[0]
        cls_idx = np.argmax(prob)
        score = prob[cls_idx]

        if score > best_score:
            best_score = score
            best_cls = cls_idx
            best_box = (x,y,x+w,y+h)

    x1,y1,x2,y2 = best_box
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    txt = f"{CLASSES[best_cls]}:{best_score:.2f}"
    cv2.putText(img,txt,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("result-SS", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "AgroPest-12" / "valid" / "images" / "bees-7-_jpg.rf.3a89fc091c6af809ec16edd295db4319.jpg"
    detect_image_ss(test_img)
