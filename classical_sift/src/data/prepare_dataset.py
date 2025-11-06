
import os
from pathlib import Path
from classical_sift.configs.config import FILE_PREFIX_TO_CLASS, CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "AgroPest-12"

def list_images(split="train"):
    img_dir = DATA_ROOT / split / "images"
    names = os.listdir(img_dir)
    selected = []
    for fname in names:
        low = fname.lower()
        for prefix, canonical in FILE_PREFIX_TO_CLASS.items():
            if low.startswith(prefix + "-"):
                if canonical in CLASSES:
                    selected.append((canonical, img_dir / fname))
                break
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

