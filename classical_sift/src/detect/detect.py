import cv2
import pickle
import numpy as np
from pathlib import Path

from classical_sift.src.detect.window_detect import window_detect
from classical_sift.configs.config import CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = PROJECT_ROOT / "classical_sift" / "src" / "vocab" / "vocab_k200.pkl"
MODEL_PATH = PROJECT_ROOT / "classical_sift" / "src" / "models" / "svm_bow_model.pkl"


def detect_image(img_path):
    img = cv2.imread(str(img_path))
    with open(VOCAB_PATH, "rb") as f:
        kmeans = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        svm = pickle.load(f)

    cls_idx, score, bbox = window_detect(img, kmeans, svm)

    x1,y1,x2,y2 = bbox
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    text = f"{CLASSES[cls_idx]}:{score:.2f}"
    cv2.putText(img,text,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "AgroPest-12" / "valid" / "images" / "bees-7-_jpg.rf.3a89fc091c6af809ec16edd295db4319.jpg"
    detect_image(test_img)
