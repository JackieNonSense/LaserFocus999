This folder is only the YOLOv8s branch of our group project and is not the full submission.

\# COMP9517 Project – YOLOv8s Code



This folder contains the code and configuration for the \*\*YOLOv8s\*\* branch of our group project \*“Insect Detection and Classification: A Comparative Study of Machine Learning and Deep Learning Approaches”\*.



The notebook reproduces the experiments reported in the YOLOv8s sections of the report.



---



\## 1. Folder Structure



```text

.

├── archive/

│   ├── data.yaml          # YOLO dataset config (train / val / test)

│   ├── knowledge.json     # Notes about the cleaned AgroPest-12 dataset

│   ├── train/             # Training images and labels  (YOLO format)\*

│   ├── valid/             # Validation images and labels\*

│   └── test/              # Test images and labels\*

├── ass1ver2.ipynb         # Main notebook for YOLOv8s

├── yolov8s\_val\_summary.csv    # Overall validation metrics (mAP, Precision, Recall)

└── yolov8s\_val\_per\_class\_AP.csv  # Per-class AP on the validation set

\* The train/, valid/ and test/ folders are expected to follow the standard YOLO structure:

train/

&nbsp; images/

&nbsp; labels/

valid/

&nbsp; images/

&nbsp; labels/

test/

&nbsp; images/

&nbsp; labels/

2\. Environment



The code was tested with:



Python 3.10



CUDA GPU (NVIDIA RTX 4060 Laptop)



Main Python packages:



torch and torchvision



ultralytics



numpy



pandas



matplotlib



IPython



A minimal installation can be done with:

pip install torch torchvision ultralytics numpy pandas matplotlib ipykernel



3\. How to Run the Notebook



Open ass1ver2.ipynb in Jupyter Notebook, JupyterLab, or VS Code.



Make sure the working directory is the project root (the folder that contains archive/ and ass1ver2.ipynb).

The notebook assumes that:



the dataset config is at archive/data.yaml;



the images and labels are inside archive/train, archive/valid, and archive/test.



Run all cells in order:



Reproducibility \& imports

Sets the random seed, checks the CUDA device, and imports YOLO from ultralytics.



Load base YOLOv8s model

Loads yolov8s.pt (COCO-pretrained) as the starting point for our detector.



Training cell

Calls model.train() with:



data="archive/data.yaml"



epochs=50



imgsz=960



batch=8



device=0



project="AgroPest", name="YOLOv8s"



patience=50, plots=True

This cell is the most time-consuming. On a laptop GPU it can take around 1–2 hours.

For marking, it is acceptable to skip re-running this cell and only inspect the saved outputs in the notebook.



Display training curves

Displays results.png (training and validation losses, precision, recall, mAP).

These plots are used in the “Training Behaviour” slide of the presentation.



Load best weights and run validation

Finds best.pt from the training run and calls model.val() on the validation set.

Validation metrics are saved into a Python object and also into:



yolov8s\_val\_summary.csv



yolov8s\_val\_per\_class\_AP.csv



Generate predictions on validation images

Uses model.predict(source="archive/valid/images", save=True) to create annotated prediction images under runs/detect/val.

A helper cell shows a few sample predictions inside the notebook.



After running, you can inspect:



Training curves (loss, precision, recall, mAP)



Per-class AP values from yolov8s\_val\_per\_class\_AP.csv



Overall metrics from yolov8s\_val\_summary.csv



Example detection images in runs/detect/val/



4\. Notes for Markers



The notebook has been fully run once, so all main figures (training curves, confusion matrix, metrics tables) are already embedded in the .ipynb file.



Re-running the heavy training cell is not required for marking; opening the notebook and running light cells (e.g., display or analysis) is enough to verify the workflow.



The code is version-robust where possible (for example when reading metrics attributes), so it should work with minor updates of the ultralytics package.



