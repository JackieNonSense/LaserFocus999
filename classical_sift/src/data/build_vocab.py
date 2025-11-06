import random
import pickle
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from classical_sift.src.data.prepare_dataset import list_images
from classical_sift.src.data.feature_extract import sift

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = PROJECT_ROOT / "classical_sift" / "src" / "vocab" / "vocab_k200.pkl"

K = 200  # chosen cluster size

def collect_descriptors(split="train", limit=300):
    """collect raw descriptors from random images to build vocab"""
    data = list_images(split)
    random.shuffle(data)
    descs = []
    count = 0

    for cls, img_path in data:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, d = sift.detectAndCompute(img, None)
        if d is not None:
            descs.append(d)
            count += 1
        if limit and count >= limit:
            break

    descs = np.vstack(descs)
    return descs

if __name__ == "__main__":
    import cv2
    
    print("Collecting descriptors...")
    descs = collect_descriptors("train", limit=300)
    print("Collected descriptor shape:", descs.shape)

    print(f"Clustering into {K} words...")
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=K*10, verbose=1)
    kmeans.fit(descs)

    print("Saving vocab...")
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(kmeans, f)

    print("Done. vocab saved to:", VOCAB_PATH)
