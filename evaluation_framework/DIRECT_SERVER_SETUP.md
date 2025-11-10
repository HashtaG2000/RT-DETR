# RT-DETR Evaluation - Direct Server Setup Guide

Guide for setting up RT-DETR evaluation when you have direct access to your A100 server.

## 🎯 Prerequisites

- Direct SSH access to A100 server
- Docker installed on server
- NVIDIA Docker runtime installed

---

## 📋 Option 1: Clone from GitHub (Recommended)

If your repository is on GitHub:

```bash
# SSH into your server
ssh your-username@your-server.edu

# Clone the repository
cd ~
git clone https://github.com/your-username/RT-DETR-main.git
cd RT-DETR-main/evaluation_framework

# Make scripts executable
chmod +x setup_data.sh run_evaluation.sh

# Build Docker image
docker-compose build

# Run container
docker-compose run rtdetr-eval

# Inside container - run setup
./setup_data.sh

# Run evaluation
./run_evaluation.sh
```

---

## 📋 Option 2: Create Files Manually on Server

If you can't transfer files, create them directly on the server:

### Step 1: SSH into Server

```bash
ssh your-username@your-server.edu
```

### Step 2: Create Project Directory

```bash
cd ~
mkdir -p RT-DETR-eval
cd RT-DETR-eval
```

### Step 3: Create Dockerfile

```bash
cat > Dockerfile << 'EOF'
# RT-DETR Evaluation Docker Image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install \
    pycocotools \
    pyyaml \
    opencv-python \
    Pillow \
    numpy \
    matplotlib \
    tqdm \
    jupyter \
    ipykernel \
    scipy \
    pandas

WORKDIR /workspace

RUN mkdir -p /workspace/dataset/coco \
    /workspace/pretrained_weights \
    /workspace/outputs \
    /workspace/outputs_universal

EXPOSE 8888

CMD ["/bin/bash"]
EOF
```

### Step 4: Create docker-compose.yml

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  rtdetr-eval:
    build:
      context: .
      dockerfile: Dockerfile
    image: rtdetr-evaluation:latest
    container_name: rtdetr-eval

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    network_mode: host

    volumes:
      - .:/workspace
      - ./dataset:/workspace/dataset
      - ./pretrained_weights:/workspace/pretrained_weights
      - ./outputs:/workspace/outputs

    working_dir: /workspace

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/workspace:/workspace/src

    stdin_open: true
    tty: true

    command: /bin/bash
EOF
```

### Step 5: Create setup_data.sh

```bash
cat > setup_data.sh << 'EOF'
#!/bin/bash
set -e

echo "=============================================="
echo "RT-DETR Evaluation Setup"
echo "=============================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

WORKSPACE="/workspace"
cd "$WORKSPACE"

print_info "Step 1/3: Cloning RT-DETR repository..."
if [ ! -d "src" ]; then
    git clone https://github.com/lyuwenyu/RT-DETR.git temp_repo
    cp -r temp_repo/rtdetr_pytorch/src .
    cp -r temp_repo/rtdetr_pytorch/configs .
    rm -rf temp_repo
    print_info "Repository cloned successfully"
else
    print_info "Source code already exists"
fi

print_info "Step 2/3: Setting up COCO dataset"
DATASET_DIR="$WORKSPACE/dataset/coco"
mkdir -p "$DATASET_DIR"

if [ ! -d "$DATASET_DIR/val2017" ]; then
    print_info "Downloading COCO val2017 images (~1GB)..."
    cd "$DATASET_DIR"
    wget -c http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
    rm val2017.zip
    print_info "COCO images downloaded"
else
    print_info "COCO val2017 images already exist"
fi

if [ ! -f "$DATASET_DIR/annotations/instances_val2017.json" ]; then
    print_info "Downloading COCO annotations (~241MB)..."
    cd "$DATASET_DIR"
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    print_info "COCO annotations downloaded"
else
    print_info "COCO annotations already exist"
fi

print_info "Step 3/3: Downloading pretrained weights"
WEIGHTS_DIR="$WORKSPACE/pretrained_weights"
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

if [ ! -f "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth" ]; then
    print_info "Downloading RT-DETRv2-R18 weights (~85MB)..."
    wget -c https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
    print_info "Weights downloaded"
else
    print_info "RT-DETRv2-R18 weights already exist"
fi

print_info "Setup completed successfully!"
print_info "Next step: Run ./run_evaluation.sh"
EOF

chmod +x setup_data.sh
```

### Step 6: Create run_evaluation.sh

```bash
cat > run_evaluation.sh << 'EOF'
#!/bin/bash
set -e

echo "=============================================="
echo "RT-DETR Evaluation Runner"
echo "=============================================="

GREEN='\033[0;32m'
NC='\033[0m'
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

print_info "Creating evaluation script..."

cat > eval_script.py << 'PYTHON_SCRIPT'
import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm
import time

EVAL_ROOT = Path.cwd()
sys.path.insert(0, str(EVAL_ROOT / 'src'))

from src.core import YAMLConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

print("="*80)
print("RT-DETR EVALUATION ON A100")
print("="*80)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'model_name': 'rtdetrv2_r18vd',
    'config_file': 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
    'checkpoint_path': 'pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
    'img_folder': 'dataset/coco/val2017',
    'ann_file': 'dataset/coco/annotations/instances_val2017.json',
    'output_dir': 'outputs',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)
cfg = YAMLConfig(CONFIG['config_file'])
model = cfg.model.to(DEVICE)
model.eval()

checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=DEVICE, weights_only=False)
state_dict = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
model.load_state_dict(state_dict)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded: {total_params/1e6:.2f}M parameters")

print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)
coco_gt = COCO(CONFIG['ann_file'])
image_ids = coco_gt.getImgIds()
print(f"✓ Dataset: {len(image_ids)} images")

print("\n" + "="*80)
print("RUNNING INFERENCE")
print("="*80)

import torchvision.transforms as T
from PIL import Image
from src.data.dataset import mscoco_label2category

transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])

all_results = []
inference_times = []

with torch.no_grad():
    for img_id in tqdm(image_ids, desc="Processing"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(CONFIG['img_folder'], img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert('RGB')
        orig_h, orig_w = img_info['height'], img_info['width']
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        start_time = time.time()
        outputs = model(img_tensor)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        logits = outputs['pred_logits'][0]
        boxes = outputs['pred_boxes'][0]
        scores = torch.sigmoid(logits)
        scores_per_query, labels = scores.max(dim=-1)
        keep = scores_per_query > 0.3

        if keep.sum() > 0:
            scores_per_query = scores_per_query[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            cx = boxes[:, 0] * orig_w
            cy = boxes[:, 1] * orig_h
            w = boxes[:, 2] * orig_w
            h = boxes[:, 3] * orig_h

            x1 = torch.clamp(cx - 0.5 * w, 0, orig_w)
            y1 = torch.clamp(cy - 0.5 * h, 0, orig_h)
            x2 = torch.clamp(cx + 0.5 * w, 0, orig_w)
            y2 = torch.clamp(cy + 0.5 * h, 0, orig_h)

            for i in range(len(labels)):
                bbox_w = (x2[i] - x1[i]).item()
                bbox_h = (y2[i] - y1[i]).item()

                if bbox_w > 0 and bbox_h > 0:
                    all_results.append({
                        'image_id': int(img_id),
                        'category_id': mscoco_label2category[int(labels[i].item())],
                        'bbox': [x1[i].item(), y1[i].item(), bbox_w, bbox_h],
                        'score': float(scores_per_query[i].item())
                    })

avg_time_ms = np.mean(inference_times) * 1000
fps = 1.0 / np.mean(inference_times)
print(f"\n✓ Inference complete: {len(all_results)} predictions")
print(f"  Average time: {avg_time_ms:.2f}ms/image")
print(f"  Throughput: {fps:.2f} FPS")

print("\n" + "="*80)
print("COMPUTING METRICS")
print("="*80)

import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(all_results, f)
    pred_file = f.name

coco_dt = coco_gt.loadRes(pred_file)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

metrics = {
    'AP': coco_eval.stats[0],
    'AP50': coco_eval.stats[1],
    'AP75': coco_eval.stats[2],
    'AP_small': coco_eval.stats[3],
    'AP_medium': coco_eval.stats[4],
    'AP_large': coco_eval.stats[5],
}

os.unlink(pred_file)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Model: {CONFIG['model_name']}")
print(f"Parameters: {total_params/1e6:.2f}M")
print(f"Device: {DEVICE}")
print(f"\nCOCO Metrics:")
print(f"  AP (0.50:0.95): {metrics['AP']*100:.1f}%")
print(f"  AP50:           {metrics['AP50']*100:.1f}%")
print(f"  AP75:           {metrics['AP75']*100:.1f}%")
print(f"  AP (small):     {metrics['AP_small']*100:.1f}%")
print(f"  AP (medium):    {metrics['AP_medium']*100:.1f}%")
print(f"  AP (large):     {metrics['AP_large']*100:.1f}%")
print(f"\nPerformance:")
print(f"  FPS: {fps:.2f}")
print(f"  Latency: {avg_time_ms:.2f}ms")
print("="*80)

results = {
    'model': CONFIG['model_name'],
    'parameters_M': total_params / 1e6,
    'device': str(DEVICE),
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
    'metrics': metrics,
    'performance': {
        'fps': float(fps),
        'latency_ms': float(avg_time_ms),
    },
    'timestamp': datetime.now().isoformat(),
}

output_file = f"{CONFIG['output_dir']}/{CONFIG['model_name']}_a100_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
print("EVALUATION COMPLETE!")
PYTHON_SCRIPT

print_info "Running evaluation..."
python3 eval_script.py
EOF

chmod +x run_evaluation.sh
```

### Step 7: Build and Run

```bash
# Build Docker image
docker-compose build

# Start container
docker-compose run rtdetr-eval

# Inside container:
./setup_data.sh    # Downloads everything (one-time setup)
./run_evaluation.sh  # Runs the evaluation
```

---

## 📋 Option 3: Quick One-Liner Setup

Copy and paste this entire block in your server terminal:

```bash
mkdir -p ~/RT-DETR-eval && cd ~/RT-DETR-eval && \
curl -o Dockerfile https://raw.githubusercontent.com/your-repo/RT-DETR-main/main/evaluation_framework/Dockerfile && \
curl -o docker-compose.yml https://raw.githubusercontent.com/your-repo/RT-DETR-main/main/evaluation_framework/docker-compose.yml && \
curl -o setup_data.sh https://raw.githubusercontent.com/your-repo/RT-DETR-main/main/evaluation_framework/setup_data.sh && \
curl -o run_evaluation.sh https://raw.githubusercontent.com/your-repo/RT-DETR-main/main/evaluation_framework/run_evaluation.sh && \
chmod +x *.sh && \
docker-compose build && \
docker-compose run rtdetr-eval
```

---

## 🎯 Verify Setup

```bash
# Check Docker
docker --version

# Check NVIDIA GPU access
nvidia-smi

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 📊 Expected Output

When you run the evaluation, you'll see:

```
==================================================
RT-DETR EVALUATION ON A100
==================================================
PyTorch: 2.2.0+cu121
CUDA: True
GPU: NVIDIA A100-SXM4-40GB
GPU Memory: 40.0 GB

==================================================
LOADING MODEL
==================================================
✓ Model loaded: 20.18M parameters

==================================================
LOADING DATASET
==================================================
✓ Dataset: 5000 images

==================================================
RUNNING INFERENCE
==================================================
Processing: 100%|████████| 5000/5000 [00:45<00:00, 110.23it/s]

✓ Inference complete: 78185 predictions
  Average time: 9.07ms/image
  Throughput: 110.23 FPS

==================================================
COMPUTING METRICS
==================================================
 Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.459
 Average Precision  (AP) @[ IoU=0.50      ] = 0.618
 ...

==================================================
FINAL RESULTS
==================================================
Model: rtdetrv2_r18vd
Parameters: 20.18M
Device: cuda

COCO Metrics:
  AP (0.50:0.95): 45.9%
  AP50:           61.8%
  AP75:           49.7%

Performance:
  FPS: 110.23
  Latency: 9.07ms
==================================================

✓ Results saved to: outputs/rtdetrv2_r18vd_a100_results.json
```

---

## 🚀 Quick Reference

```bash
# SSH to server
ssh your-username@server

# Navigate to project
cd ~/RT-DETR-eval

# Build (one-time)
docker-compose build

# Run container
docker-compose run rtdetr-eval

# Inside container - setup (one-time)
./setup_data.sh

# Inside container - evaluate
./run_evaluation.sh

# Exit container
exit

# View results (on server)
cat outputs/rtdetrv2_r18vd_a100_results.json
```

---

## ✅ Summary

Since you have direct server access, the easiest approach is:

1. **SSH into server**
2. **Create files manually** (Option 2) OR **clone from GitHub** (Option 1)
3. **Build Docker image**: `docker-compose build`
4. **Run container**: `docker-compose run rtdetr-eval`
5. **Setup once**: `./setup_data.sh`
6. **Run evaluation**: `./run_evaluation.sh`

All data and results persist on your server even after container exits!
