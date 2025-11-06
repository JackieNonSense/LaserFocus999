
import cv2
import numpy as np
from classical_sift.src.data.prepare_dataset import list_images

sift = cv2.SIFT_create()

def extract_feature(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    keypoints, desc = sift.detectAndCompute(img, None)
    if desc is None:
        # 没检测到特征点就用零向量
        return np.zeros(128)
    # mean pooling 把 N x 128 → 128
    return desc.mean(axis=0)

if __name__ == "__main__":
    train_list = list_images("train")
    X = []
    y = []
    label_map = {"Bees":0, "Beetles":1, "Weevils":2}

    for cls, img_path in train_list[:50]: # 先只抽50张测试
        feat = extract_feature(img_path)
        if feat is not None:
            X.append(feat)
            y.append(label_map[cls])

    print("X shape:", np.array(X).shape)
    print("y length:", len(y))