import random
import pickle
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import cv2

from classical_sift.src.data.prepare_dataset import list_images
from classical_sift.src.data.feature_extract import sift

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = PROJECT_ROOT / "classical_sift" / "src" / "vocab" / "vocab_k200.pkl"

K = 200  # chosen cluster size
MAX_IMAGES = 2000  # ~25% of the dataset, which is a standard sampling strategy in BOW pipelines

def build_vocab():
    data = list_images("train")
    random.shuffle(data)

    # full SIFT memory overflow
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=K*10, verbose=1)

    processed = 0
    for cls, img_path in data[:MAX_IMAGES]:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, d = sift.detectAndCompute(img, None)
        if d is not None:
            kmeans.partial_fit(d) 
            processed += 1

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(kmeans, f)

    print("Done. vocab saved:", VOCAB_PATH)

if __name__=="__main__":
    MAX_IMAGES = 2000
    build_vocab()


# def collect_descriptors(split="train"):
#     data = list_images(split)
#     random.shuffle(data)
#     descs = []
#     for cls, img_path in data[:MAX_IMAGES]:
#         img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             continue
#         _, d = sift.detectAndCompute(img, None)
#         if d is not None:
#             descs.append(d)

#     descs = np.vstack(descs)
#     return descs

# if __name__ == "__main__":
#     import cv2
    
#     print("Collecting descriptors...")
#     descs = collect_descriptors("train")
#     print("Collected descriptor shape:", descs.shape)

#     print(f"Clustering into {K} words...")
#     kmeans = MiniBatchKMeans(n_clusters=K, batch_size=K*10, verbose=1)
#     kmeans.fit(descs)

#     print("Saving vocab...")
#     with open(VOCAB_PATH, "wb") as f:
#         pickle.dump(kmeans, f)

#     print("Done. vocab saved to:", VOCAB_PATH)
