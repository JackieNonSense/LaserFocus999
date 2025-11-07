import cv2

def generate_windows(img, win_ratio=0.25, stride_ratio=0.125):
    """
    img: numpy ndarray (BGR)
    win_ratio: window size relative to image width  (1/4 by default)
    stride_ratio: step size relative to window width (half-window stride by default)

    return: window list:  [(x1, y1, x2, y2), ...]
    """
    h, w = img.shape[:2]
    win_w = int(w * win_ratio)
    win_h = win_w  # square window baseline

    step = int(win_w * stride_ratio)

    windows = []
    for y in range(0, h - win_h, step):
        for x in range(0, w - win_w, step):
            windows.append((x, y, x + win_w, y + win_h))

    return windows


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread("../../../data/AgroPest-12/train/images/bees-1-_jpg.rf.7ccf6d0124e82a6cf5948de1a3e4f02b.jpg")
    windows = generate_windows(img)

    print("total windows:", len(windows))
