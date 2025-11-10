# Universal Object Detection Evaluation Framework

A flexible evaluation framework that works with **any** object detection model trained on COCO.

## Supported Models

- ✅ **RT-DETR** (all variants: R18, R50, R101, v1, v2, v3)
- ✅ **YOLO** (v5, v8, v11 via Ultralytics)
- ✅ **Torchvision models** (Faster R-CNN, RetinaNet, FCOS, etc.)
- ✅ **Custom models** (extend `BaseModelAdapter`)

## Quick Start

### 1. Evaluate RT-DETR Models

```bash
# RT-DETRv2-R18
python universal_evaluate.py \
    --model-type rtdetr \
    --config configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
    --checkpoint ../pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth

# RT-DETRv2-R50
python universal_evaluate.py \
    --model-type rtdetr \
    --config configs/rtdetrv2/rtdetrv2_r50vd_120e_coco.yml \
    --checkpoint ../pretrained_weights/rtdetrv2_r50vd_120e_coco.pth
```

### 2. Evaluate YOLO Models

```bash
# Install ultralytics first
pip install ultralytics

# YOLOv8n
python universal_evaluate.py \
    --model-type yolo \
    --model-path yolov8n.pt

# YOLOv11m
python universal_evaluate.py \
    --model-type yolo \
    --model-path yolo11m.pt
```

### 3. Evaluate Torchvision Models

```python
# Use Python script or notebook
from model_adapters import create_model_adapter
import torchvision

# Load Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
adapter = create_model_adapter('torchvision', model=model, device='cuda')

# Run evaluation (see universal_evaluate.py for full example)
```

## Advanced Usage

### Evaluate on Subset

```bash
# Evaluate on first 100 images only
python universal_evaluate.py \
    --model-type rtdetr \
    --config configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
    --checkpoint weights/rtdetr.pth \
    --num-images 100
```

### Skip FPS Benchmarking

```bash
python universal_evaluate.py \
    --model-type yolo \
    --model-path yolov8n.pt \
    --skip-fps
```

### Custom Dataset Path

```bash
python universal_evaluate.py \
    --model-type rtdetr \
    --config configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
    --checkpoint weights/rtdetr.pth \
    --img-folder /path/to/coco/val2017 \
    --ann-file /path/to/instances_val2017.json
```

## Using in Jupyter Notebook

See `RTDETR_Evaluation.ipynb` for an example. To make it work with any model:

```python
from model_adapters import create_model_adapter

# Option 1: RT-DETR
adapter = create_model_adapter(
    'rtdetr',
    config_path='configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
    checkpoint_path='../pretrained_weights/rtdetrv2_r18vd.pth',
    device='cuda'
)

# Option 2: YOLO
adapter = create_model_adapter(
    'yolo',
    model_path='yolov8n.pt',
    device='cuda'
)

# Option 3: Torchvision
import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
adapter = create_model_adapter(
    'torchvision',
    model=model,
    model_name='Faster R-CNN R50',
    device='cuda'
)

# Then use adapter.predict(image, orig_size) for inference
```

## Adding Custom Models

Create a new adapter by extending `BaseModelAdapter`:

```python
from model_adapters import BaseModelAdapter

class MyCustomAdapter(BaseModelAdapter):
    def __init__(self, model_path, device='cuda'):
        # Load your model
        model = load_my_model(model_path)
        super().__init__(model, device)

    def preprocess(self, image):
        # Convert PIL image to model input format
        return preprocess_image(image)

    def predict(self, image, orig_size):
        # Run inference and return predictions
        img_tensor = self.preprocess(image)
        outputs = self.model(img_tensor)

        # Convert to COCO format
        predictions = []
        for box, score, label in zip(outputs.boxes, outputs.scores, outputs.labels):
            predictions.append({
                'bbox': [x1, y1, width, height],
                'score': float(score),
                'label': int(label),
                'category_id': int(label) + 1  # Adjust as needed
            })
        return predictions

    def get_model_info(self):
        return {
            'model_type': 'MyCustomModel',
            'model_path': self.model_path,
        }
```

## Output Format

Results are saved in JSON format:

```json
{
  "model_info": {
    "model_type": "RT-DETR",
    "parameters_M": 21.96
  },
  "evaluation": {
    "AP": 0.459,
    "AP50": 0.618,
    "AP75": 0.497,
    "fps": 31.96
  },
  "fps_benchmark": {
    "fps": 31.96,
    "latency_ms": 31.29
  }
}
```

## Architecture

The framework uses an **adapter pattern**:

```
┌─────────────────────┐
│   Your Model        │
│ (RT-DETR/YOLO/etc)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Model Adapter     │  ← Converts model-specific format
│  - preprocess()     │     to universal format
│  - predict()        │
│  - get_model_info() │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Universal Evaluator │  ← Same evaluation code
│  - COCO metrics     │     for all models
│  - FPS benchmark    │
└─────────────────────┘
```

## Requirements

```bash
# Core requirements
pip install torch torchvision
pip install pycocotools
pip install Pillow numpy tqdm

# For RT-DETR (already have)
# Your existing RT-DETR dependencies

# For YOLO (optional)
pip install ultralytics

# For visualization
pip install matplotlib
```

## Troubleshooting

### CUDA out of memory
- Reduce `--num-images` to evaluate on fewer images
- Use `--device cpu` for CPU evaluation

### Model-specific issues
- **RT-DETR**: Ensure config file matches checkpoint
- **YOLO**: Make sure ultralytics is installed
- **Torchvision**: Models must be in eval mode

### Category ID mismatch
- The adapter automatically handles COCO category mapping
- RT-DETR uses `mscoco_label2category` mapping
- YOLO uses label + 1
- Torchvision models already use COCO IDs

## Performance Comparison

Example results on RTX 4060 Laptop GPU:

| Model | AP | AP50 | FPS | Latency |
|-------|-------|------|-----|---------|
| RT-DETRv2-R18 | 45.9% | 61.8% | 32 | 31ms |
| RT-DETRv2-R50 | 53.0% | 71.0% | 25 | 40ms |
| YOLOv8n | 37.3% | 52.6% | 80 | 12ms |
| YOLOv8m | 50.2% | 67.2% | 45 | 22ms |

## License

Same as parent RT-DETR project.
