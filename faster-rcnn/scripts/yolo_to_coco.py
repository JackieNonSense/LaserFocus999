"""
Convert YOLO format annotations to COCO format for Detectron2.

YOLO format (one .txt file per image):
    class_id center_x center_y width height (all normalized 0-1)

COCO format (single JSON file):
    {
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO bbox to COCO bbox format.

    Args:
        yolo_bbox: [center_x, center_y, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        [x, y, width, height] in pixels (COCO format: top-left corner)
    """
    center_x, center_y, w, h = yolo_bbox

    # Convert from normalized to pixel coordinates
    center_x *= img_width
    center_y *= img_height
    w *= img_width
    h *= img_height

    # Convert from center to top-left corner
    x = center_x - w / 2
    y = center_y - h / 2

    return [x, y, w, h]


def convert_yolo_to_coco(yolo_dir, split='train', class_names=None):
    """
    Convert YOLO format dataset to COCO format.

    Args:
        yolo_dir: Path to YOLO dataset (contains train/valid/test folders)
        split: 'train', 'valid', or 'test'
        class_names: List of class names (if None, will use generic names)

    Returns:
        Dictionary in COCO format
    """
    images_dir = Path(yolo_dir) / split / 'images'
    labels_dir = Path(yolo_dir) / split / 'labels'

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    # Initialize COCO format structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Get all image files
    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))

    if len(image_files) == 0:
        raise ValueError(f"No image files found in {images_dir}")

    print(f"Found {len(image_files)} images in {split} set")

    # Determine number of classes from labels
    if class_names is None:
        # Scan all label files to find max class_id
        max_class_id = -1
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            max_class_id = max(max_class_id, class_id)

        # Create generic class names
        num_classes = max_class_id + 1
        class_names = [f"class_{i}" for i in range(num_classes)]
        print(f"Detected {num_classes} classes")

    # Create categories
    for idx, class_name in enumerate(class_names):
        coco_data["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "insect"
        })

    annotation_id = 0

    # Process each image
    for img_id, img_file in enumerate(tqdm(image_files, desc=f"Converting {split}")):
        # Get image dimensions
        try:
            img = Image.open(img_file)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Warning: Could not open image {img_file}: {e}")
            continue

        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": img_width,
            "height": img_height
        })

        # Read corresponding label file
        label_file = labels_dir / f"{img_file.stem}.txt"

        if not label_file.exists():
            print(f"Warning: No label file for {img_file.name}")
            continue

        # Parse YOLO annotations
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: Invalid line in {label_file}: {line}")
                    continue

                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]

                # Convert to COCO format
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)

                # Calculate area
                area = coco_bbox[2] * coco_bbox[3]

                # Add annotation
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0
                })

                annotation_id += 1

    print(f"Converted {len(coco_data['images'])} images with {len(coco_data['annotations'])} annotations")

    return coco_data


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO format to COCO format')
    parser.add_argument('--yolo_dir', type=str, required=True,
                       help='Path to YOLO dataset directory (contains train/valid/test)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for COCO format annotations')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'valid', 'test'],
                       help='Splits to convert (default: train valid test)')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='List of class names (optional)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each split
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Converting {split} split...")
        print(f"{'='*60}")

        try:
            coco_data = convert_yolo_to_coco(
                args.yolo_dir,
                split=split,
                class_names=args.class_names
            )

            # Save to JSON file
            output_file = output_dir / f"{split}_coco.json"
            with open(output_file, 'w') as f:
                json.dump(coco_data, f, indent=2)

            print(f"Saved to {output_file}")

        except Exception as e:
            print(f"Error converting {split}: {e}")

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
