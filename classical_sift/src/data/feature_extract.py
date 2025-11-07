
import cv2
import numpy as np

sift = cv2.SIFT_create()

def extract_feature(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    keypoints, desc = sift.detectAndCompute(img, None)
    if desc is None:
        return np.zeros(128)    # mean pooling , N x 128 → 128
    return desc.mean(axis=0)
