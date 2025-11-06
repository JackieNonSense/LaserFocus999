
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "AgroPest-12"

CLASSES = ["bees", "beetle", "weevil"]

def list_images(split="train"):
    img_dir = DATA_ROOT / split / "images"
    all_imgs = os.listdir(img_dir)
    selected = []
    for cls in CLASSES:
        for name in all_imgs:
            # YOLO dataset format, like: bees-1-xxxx
            if cls.lower() in name.lower():
                selected.append((cls, img_dir/name))
    return selected

# if __name__ == "__main__":
#     train_list = list_images("train")
#     valid_list = list_images("valid")

#     print("train samples:", len(train_list))  # 1089
#     print("valid samples:", len(valid_list))  # 96

#     print("example:", train_list[:5])

if __name__ == "__main__":
    from collections import Counter
    train_list = list_images("train")
    valid_list = list_images("valid")

    print("Train class counts:", Counter([c for c,_ in train_list]))
    print("Valid class counts:", Counter([c for c,_ in valid_list]))

