#!/bin/bash

# RT-DETR Evaluation Runner Script
# Converts notebook to script and executes evaluation

set -e  # Exit on error

echo "=============================================="
echo "RT-DETR Evaluation Runner"
echo "=============================================="

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Configuration
MODEL_VARIANT=${1:-"r18"}  # Default to R18
NOTEBOOK="RTDETR_Evaluation.ipynb"
SCRIPT="RTDETR_Evaluation_Script.py"

print_info "Model variant: RT-DETRv2-${MODEL_VARIANT^^}"

# Check if notebook exists
if [ ! -f "$NOTEBOOK" ]; then
    print_warn "Notebook not found, will run existing Python script"
else
    print_info "Converting notebook to Python script..."
    jupyter nbconvert --to python "$NOTEBOOK" --output "${SCRIPT%.py}"
fi

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    print_warn "Python script not found. Creating from notebook cells..."

    cat > "$SCRIPT" << 'PYTHON_SCRIPT'
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm
import time

# Add src to path
EVAL_ROOT = Path.cwd()
sys.path.insert(0, str(EVAL_ROOT / 'src'))

from src.core import YAMLConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

print("Starting RT-DETR Evaluation")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration
MODEL_CONFIG = {
    'model_name': 'rtdetrv2_r18vd',
    'config_file': 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
    'checkpoint_path': '../pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
}

CONFIG = {
    'model_name': MODEL_CONFIG['model_name'],
    'config_file': MODEL_CONFIG['config_file'],
    'checkpoint_path': MODEL_CONFIG['checkpoint_path'],
    'img_folder': 'dataset/coco/val2017',
    'ann_file': 'dataset/coco/annotations/instances_val2017.json',
    'batch_size': 32,
    'num_workers': 1,
    'input_size': 640,
    'fps_warmup': 10,
    'fps_iterations': 100,
    'output_dir': 'outputs',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print("="*80)
print("Loading model...")
config_path = str(EVAL_ROOT / CONFIG['config_file'])
cfg = YAMLConfig(config_path)
model = cfg.model.to(DEVICE)
postprocessor = cfg.postprocessor.to(DEVICE)
model.eval()
postprocessor.eval()

checkpoint_path = str(EVAL_ROOT / CONFIG['checkpoint_path'])
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

if 'ema' in checkpoint:
    state_dict = checkpoint['ema']['module']
else:
    state_dict = checkpoint['model']

model.load_state_dict(state_dict)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {total_params/1e6:.2f}M parameters")

print("="*80)
print("Loading COCO dataset...")
coco_gt = COCO(CONFIG['ann_file'])
image_ids = coco_gt.getImgIds()
print(f"Dataset loaded: {len(image_ids)} images")

print("="*80)
print("Running inference...")

import torchvision.transforms as T
from PIL import Image
from src.data.dataset import mscoco_label2category

transform = T.Compose([
    T.Resize((CONFIG['input_size'], CONFIG['input_size'])),
    T.ToTensor(),
])

all_results = []
inference_times = []

with torch.no_grad():
    for idx, img_id in enumerate(tqdm(image_ids, desc="Inference")):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(CONFIG['img_folder'], img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_h, orig_w = img_info['height'], img_info['width']

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        start_time = time.time()
        outputs = model(img_tensor)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        logits = outputs['pred_logits'][0]
        boxes = outputs['pred_boxes'][0]

        scores = torch.sigmoid(logits)
        scores_per_query, labels = scores.max(dim=-1)

        keep = scores_per_query > 0.3
        if keep.sum() == 0:
            continue

        scores_per_query = scores_per_query[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        cx = boxes[:, 0] * orig_w
        cy = boxes[:, 1] * orig_h
        w = boxes[:, 2] * orig_w
        h = boxes[:, 3] * orig_h

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        x1 = torch.clamp(x1, 0, orig_w)
        y1 = torch.clamp(y1, 0, orig_h)
        x2 = torch.clamp(x2, 0, orig_w)
        y2 = torch.clamp(y2, 0, orig_h)

        for i in range(len(labels)):
            bbox_x1 = x1[i].item()
            bbox_y1 = y1[i].item()
            bbox_w = (x2[i] - x1[i]).item()
            bbox_h = (y2[i] - y1[i]).item()

            if bbox_w > 0 and bbox_h > 0:
                model_label = int(labels[i].item())
                coco_category_id = mscoco_label2category[model_label]

                all_results.append({
                    'image_id': int(img_id),
                    'category_id': coco_category_id,
                    'bbox': [bbox_x1, bbox_y1, bbox_w, bbox_h],
                    'score': float(scores_per_query[i].item())
                })

avg_time = np.mean(inference_times) * 1000
print(f"Inference complete: {len(all_results)} predictions, {avg_time:.2f}ms avg")

print("="*80)
print("Computing COCO metrics...")

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
    'AR': coco_eval.stats[6],
}

os.unlink(pred_file)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"AP (IoU=0.50:0.95): {metrics['AP']:.1%}")
print(f"AP50 (IoU=0.50):    {metrics['AP50']:.1%}")
print(f"AP75 (IoU=0.75):    {metrics['AP75']:.1%}")
print(f"Parameters:         {total_params/1e6:.2f}M")
print(f"FPS:                {1000/avg_time:.2f}")
print("="*60)

results_dict = {
    'model': CONFIG['model_name'],
    'parameters_M': total_params / 1e6,
    'device': DEVICE,
    'evaluation': metrics,
    'inference_ms': avg_time,
    'fps': 1000/avg_time,
    'timestamp': datetime.now().isoformat(),
}

output_json = f"{CONFIG['output_dir']}/{CONFIG['model_name']}_results.json"
with open(output_json, 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nResults saved to: {output_json}")
print("EVALUATION COMPLETE!")
PYTHON_SCRIPT
fi

# Run the evaluation
print_info "Starting evaluation..."
python3 "$SCRIPT"

print_info "Evaluation completed successfully!"
print_info "Check the 'outputs' directory for results"
