# AutoDL Server Setup Guide

Complete guide to set up AutoDL server for Faster R-CNN training.

## Step 1: Get AutoDL SSH Connection Info

1. **Login to AutoDL**: https://www.autodl.com/console/instance/list
2. **Start/Create Instance**:
   - Recommended: RTX 3090 or better
   - Image: PyTorch (with CUDA 11.x)
3. **Get SSH Info**:
   - Click on your instance
   - Find SSH connection details:
     - Host: `region-x.autodl.com` (or IP address)
     - Port: `xxxxx` (5-digit port number)
     - Username: `root`
     - Password: (shown in console)

**Note down these values!**

## Step 2: Configure VSCode SSH Connection

### 2.1 Install Remote-SSH Extension
- Open VSCode
- Extensions (Ctrl+Shift+X)
- Search "Remote - SSH"
- Install by Microsoft

### 2.2 Configure SSH Config File

**On Windows:**
1. Open PowerShell
2. Create SSH config:
```powershell
mkdir -Force $HOME\.ssh
notepad $HOME\.ssh\config
```

**Add this to config file:**
```
Host autodl-insect
    HostName region-x.autodl.com
    User root
    Port xxxxx
```

Replace:
- `region-x.autodl.com` with your actual host
- `xxxxx` with your actual port

**On Linux/Mac:**
```bash
mkdir -p ~/.ssh
nano ~/.ssh/config
```

Add the same content as above.

### 2.3 Connect from VSCode

1. Press `F1` or `Ctrl+Shift+P`
2. Type: `Remote-SSH: Connect to Host`
3. Select `autodl-insect`
4. Enter password when prompted
5. Select `Linux` as the platform
6. Wait for connection...

**Success!** You should see "SSH: autodl-insect" in bottom-left corner of VSCode.

## Step 3: Upload Dataset to AutoDL

### Method A: Using rsync (Recommended)

**On your local machine (PowerShell/Terminal):**
```bash
# Upload dataset
rsync -avz --progress F:/Desktop/AgroPest-12/ autodl-insect:~/data/AgroPest-12/

# This will upload:
# - train/images and train/labels
# - valid/images and valid/labels
# - test/images and test/labels
```

### Method B: Using scp

```bash
scp -r F:/Desktop/AgroPest-12 autodl-insect:~/data/
```

### Method C: Using VSCode (Slower)

1. Connect to AutoDL via VSCode
2. Open folder: `~/data/`
3. Right-click → Upload folder
4. Select `F:/Desktop/AgroPest-12`

**Verify upload:**
```bash
# In AutoDL terminal
ls -lh ~/data/AgroPest-12/
du -sh ~/data/AgroPest-12/
```

## Step 4: Clone Git Repository

**In AutoDL terminal:**
```bash
cd ~
git clone <your-repo-url> LaserFocus999
cd LaserFocus999
git checkout Yuchao/Fast-R-CNN-1
```

## Step 5: Setup Python Environment

### 5.1 Check CUDA Version
```bash
nvidia-smi
```

Note the CUDA version (e.g., 11.8, 12.1)

### 5.2 Create Conda Environment
```bash
cd ~/LaserFocus999/faster-rcnn

# Create environment
conda create -n insect python=3.8 -y
conda activate insect
```

### 5.3 Install PyTorch

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check: https://pytorch.org/get-started/locally/

**Verify PyTorch:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Should output:
```
PyTorch: 2.x.x
CUDA: True
```

### 5.4 Install Detectron2

```bash
# Clone and install
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

**Verify Detectron2:**
```bash
python -c "import detectron2; print('Detectron2:', detectron2.__version__)"
```

### 5.5 Install Other Dependencies

```bash
pip install -r requirements.txt
```

## Step 6: Convert Dataset Format

```bash
# Make sure you're in faster-rcnn directory
cd ~/LaserFocus999/faster-rcnn

# Convert YOLO to COCO format
python scripts/yolo_to_coco.py \
    --yolo_dir ~/data/AgroPest-12 \
    --output_dir outputs/coco_annotations \
    --splits train valid test
```

**Verify conversion:**
```bash
ls -lh outputs/coco_annotations/
# Should see: train_coco.json, valid_coco.json, test_coco.json
```

## Step 7: Test Everything

```bash
# Test dataset loading
python -c "
import sys
sys.path.insert(0, 'src')
from data.dataset import register_all_agropest_splits

register_all_agropest_splits(
    data_root='~/data/AgroPest-12',
    coco_json_dir='outputs/coco_annotations'
)

from detectron2.data import DatasetCatalog
train_data = DatasetCatalog.get('agropest_train')
print(f'Training samples: {len(train_data)}')
print('Setup successful!')
"
```

## Step 8: Start Training!

```bash
# Activate environment
conda activate insect

# Start training
python scripts/train.py \
    --config-file configs/faster_rcnn_R50_FPN.yaml \
    --data-root ~/data/AgroPest-12 \
    --coco-json-dir outputs/coco_annotations \
    --num-gpus 1
```

**Monitor training:**

In another terminal (or use tmux):
```bash
tensorboard --logdir outputs/checkpoints/faster_rcnn_R50_FPN --port 6006
```

Then forward port 6006 in VSCode:
- Click "PORTS" tab at bottom
- Forward port 6006
- Open in browser

## Troubleshooting

### SSH Connection Failed
- Check instance is running on AutoDL
- Verify host and port in SSH config
- Check password is correct

### CUDA Out of Memory
Edit `configs/faster_rcnn_R50_FPN.yaml`:
```yaml
SOLVER:
  IMS_PER_BATCH: 2  # Reduce from 4
```

### Dataset Not Found
```bash
# Check paths
ls ~/data/AgroPest-12/train/images
ls outputs/coco_annotations/
```

### Detectron2 Installation Failed
```bash
# Try pre-built version
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

## Useful Commands

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Check disk usage:**
```bash
df -h
du -sh ~/data/AgroPest-12/
```

**Keep training running after disconnect:**
```bash
# Install tmux
apt-get install tmux

# Start session
tmux new -s training

# Run training
python scripts/train.py ...

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

**Download results back to local:**
```bash
# On local machine
scp -r autodl-insect:~/LaserFocus999/faster-rcnn/outputs/results/ ./
```
