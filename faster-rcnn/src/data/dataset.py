"""
Dataset registration for Detectron2.

Register AgroPest-12 dataset in COCO format with Detectron2.
"""

import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def load_agropest_coco(json_file, image_root):
    """
    Load AgroPest dataset in COCO format.

    Args:
        json_file: Path to COCO format annotation JSON file
        image_root: Path to directory containing images

    Returns:
        List of dataset dictionaries in Detectron2 format
    """
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    # Create mapping from image_id to image info
    images = {img['id']: img for img in coco_data['images']}

    # Group annotations by image_id
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Convert to Detectron2 format
    dataset_dicts = []
    for img_id, img_info in images.items():
        record = {}

        # Image info
        record["file_name"] = os.path.join(image_root, img_info['file_name'])
        record["image_id"] = img_id
        record["height"] = img_info['height']
        record["width"] = img_info['width']

        # Annotations
        objs = []
        if img_id in img_to_anns:
            for ann in img_to_anns[img_id]:
                obj = {
                    "bbox": ann['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": ann['category_id'],
                    "iscrowd": ann.get('iscrowd', 0)
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_agropest_dataset(name, json_file, image_root, class_names=None):
    """
    Register a single split of AgroPest dataset.

    Args:
        name: Dataset name for registration (e.g., "agropest_train")
        json_file: Path to COCO JSON annotation file
        image_root: Path to images directory
        class_names: List of class names (optional)
    """
    # Register dataset
    DatasetCatalog.register(
        name,
        lambda: load_agropest_coco(json_file, image_root)
    )

    # Set metadata
    metadata_dict = {
        "json_file": json_file,
        "image_root": image_root,
        "evaluator_type": "coco"
    }

    if class_names is not None:
        metadata_dict["thing_classes"] = class_names

    MetadataCatalog.get(name).set(**metadata_dict)


def register_all_agropest_splits(data_root, coco_json_dir, splits=['train', 'valid', 'test']):
    """
    Register all splits of AgroPest dataset.

    Args:
        data_root: Root directory of AgroPest-12 dataset
        coco_json_dir: Directory containing COCO format JSON files
        splits: List of splits to register

    Example:
        data_root = "/path/to/data/AgroPest-12"
        coco_json_dir = "/path/to/faster-rcnn/outputs/coco_annotations"
        register_all_agropest_splits(data_root, coco_json_dir)
    """
    # AgroPest-12 has 12 insect classes with actual names
    class_names = [
        "Ants", "Bees", "Beetles", "Caterpillars",
        "Earthworms", "Earwigs", "Grasshoppers", "Moths",
        "Slugs", "Snails", "Wasps", "Weevils"
    ]

    for split in splits:
        # COCO JSON file
        if split == 'valid':
            json_file = os.path.join(coco_json_dir, f"valid_coco.json")
        else:
            json_file = os.path.join(coco_json_dir, f"{split}_coco.json")

        # Images directory
        image_root = os.path.join(data_root, split, 'images')

        # Register
        dataset_name = f"agropest_{split}"

        if os.path.exists(json_file) and os.path.exists(image_root):
            register_agropest_dataset(
                name=dataset_name,
                json_file=json_file,
                image_root=image_root,
                class_names=class_names
            )
            print(f"Registered dataset: {dataset_name}")
            print(f"  JSON: {json_file}")
            print(f"  Images: {image_root}")
        else:
            print(f"Warning: Could not register {dataset_name}")
            print(f"  JSON exists: {os.path.exists(json_file)}")
            print(f"  Images exist: {os.path.exists(image_root)}")


# Example usage
if __name__ == "__main__":
    # Example paths (update these for your setup)
    data_root = "../../data/AgroPest-12"
    coco_json_dir = "../outputs/coco_annotations"

    register_all_agropest_splits(data_root, coco_json_dir)

    # Test: Load a sample
    from detectron2.data import DatasetCatalog
    dataset_dicts = DatasetCatalog.get("agropest_train")
    print(f"\nLoaded {len(dataset_dicts)} training samples")
    if len(dataset_dicts) > 0:
        print("\nSample record:")
        print(dataset_dicts[0])
