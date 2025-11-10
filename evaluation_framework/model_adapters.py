"""
Universal Model Adapters for Object Detection Evaluation
Supports: RT-DETR, YOLO, Torchvision models, and custom models

Usage:
    adapter = create_model_adapter(model_type='rtdetr', config=config)
    predictions = adapter.predict(image, orig_size)
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class BaseModelAdapter(ABC):
    """Base class for all model adapters"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @abstractmethod
    def preprocess(self, image):
        """Preprocess image for model input"""
        pass

    @abstractmethod
    def predict(self, image, orig_size):
        """
        Run inference and return predictions in COCO format

        Returns:
            List of dicts with keys: 'bbox', 'score', 'label'
            bbox format: [x1, y1, width, height]
            label: class index (0-79 for COCO)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (name, params, etc)"""
        pass


class RTDETRAdapter(BaseModelAdapter):
    """Adapter for RT-DETR models"""

    def __init__(self, config_path, checkpoint_path, device='cuda'):
        from src.core import YAMLConfig
        from src.data.dataset import mscoco_label2category

        # Load config and build model
        cfg = YAMLConfig(config_path)
        model = cfg.model
        postprocessor = cfg.postprocessor

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']['module']
        else:
            state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict)

        super().__init__(model, device)
        self.postprocessor = postprocessor.to(device)
        self.postprocessor.eval()
        self.label2category = mscoco_label2category
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

    def preprocess(self, image):
        """Preprocess PIL image"""
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, image, orig_size):
        """Run RT-DETR inference"""
        img_tensor = self.preprocess(image)
        orig_h, orig_w = orig_size

        with torch.no_grad():
            outputs = self.model(img_tensor)

        # Manual postprocessing
        logits = outputs['pred_logits'][0]
        boxes = outputs['pred_boxes'][0]

        scores = torch.sigmoid(logits)
        scores_per_query, labels = scores.max(dim=-1)

        # Filter by confidence
        keep = scores_per_query > 0.3
        if keep.sum() == 0:
            return []

        scores_per_query = scores_per_query[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        # Convert normalized cxcywh to absolute xyxy
        cx = boxes[:, 0] * orig_w
        cy = boxes[:, 1] * orig_h
        w = boxes[:, 2] * orig_w
        h = boxes[:, 3] * orig_h

        x1 = torch.clamp(cx - 0.5 * w, 0, orig_w)
        y1 = torch.clamp(cy - 0.5 * h, 0, orig_h)
        x2 = torch.clamp(cx + 0.5 * w, 0, orig_w)
        y2 = torch.clamp(cy + 0.5 * h, 0, orig_h)

        # Convert to COCO format
        predictions = []
        for i in range(len(labels)):
            bbox_w = (x2[i] - x1[i]).item()
            bbox_h = (y2[i] - y1[i]).item()

            if bbox_w > 0 and bbox_h > 0:
                model_label = int(labels[i].item())
                coco_category_id = self.label2category[model_label]

                predictions.append({
                    'bbox': [x1[i].item(), y1[i].item(), bbox_w, bbox_h],
                    'score': float(scores_per_query[i].item()),
                    'label': model_label,
                    'category_id': coco_category_id
                })

        return predictions

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            'model_type': 'RT-DETR',
            'config_path': self.config_path,
            'checkpoint_path': self.checkpoint_path,
            'parameters': total_params,
            'parameters_M': total_params / 1e6,
        }


class YOLOAdapter(BaseModelAdapter):
    """Adapter for YOLO models (v5, v8, v11 via Ultralytics)"""

    def __init__(self, model_path, device='cuda'):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")

        self.yolo_model = YOLO(model_path)
        self.model_path = model_path
        # Note: YOLO handles device internally
        self.device_str = device

    def preprocess(self, image):
        """YOLO handles preprocessing internally"""
        return image

    def predict(self, image, orig_size):
        """Run YOLO inference"""
        results = self.yolo_model(image, device=self.device_str, verbose=False)[0]

        predictions = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = results.boxes.conf.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                if bbox_w > 0 and bbox_h > 0:
                    predictions.append({
                        'bbox': [float(x1), float(y1), float(bbox_w), float(bbox_h)],
                        'score': float(scores[i]),
                        'label': int(labels[i]),
                        'category_id': int(labels[i]) + 1  # COCO IDs start at 1
                    })

        return predictions

    def get_model_info(self):
        return {
            'model_type': 'YOLO',
            'model_path': self.model_path,
            'parameters': 'N/A',  # YOLO doesn't expose this easily
        }


class TorchvisionAdapter(BaseModelAdapter):
    """Adapter for Torchvision models (Faster R-CNN, RetinaNet, etc)"""

    def __init__(self, model, model_name='torchvision_model', device='cuda'):
        super().__init__(model, device)
        self.model_name = model_name

        import torchvision.transforms as T
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def preprocess(self, image):
        """Preprocess PIL image"""
        return self.transform(image).to(self.device)

    def predict(self, image, orig_size):
        """Run Torchvision model inference"""
        img_tensor = self.preprocess(image)

        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        predictions = []
        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] > 0.3:  # Confidence threshold
                x1, y1, x2, y2 = boxes[i]
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                if bbox_w > 0 and bbox_h > 0:
                    predictions.append({
                        'bbox': [float(x1), float(y1), float(bbox_w), float(bbox_h)],
                        'score': float(scores[i]),
                        'label': int(labels[i]) - 1,  # Torchvision uses 1-indexed
                        'category_id': int(labels[i])
                    })

        return predictions

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            'model_type': 'Torchvision',
            'model_name': self.model_name,
            'parameters': total_params,
            'parameters_M': total_params / 1e6,
        }


def create_model_adapter(model_type: str, **kwargs) -> BaseModelAdapter:
    """
    Factory function to create appropriate model adapter

    Args:
        model_type: One of 'rtdetr', 'yolo', 'torchvision'
        **kwargs: Model-specific arguments

    Returns:
        BaseModelAdapter instance

    Examples:
        # RT-DETR
        adapter = create_model_adapter(
            'rtdetr',
            config_path='configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
            checkpoint_path='weights/rtdetrv2_r18vd.pth',
            device='cuda'
        )

        # YOLO
        adapter = create_model_adapter(
            'yolo',
            model_path='yolov8n.pt',
            device='cuda'
        )

        # Torchvision
        import torchvision
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        adapter = create_model_adapter(
            'torchvision',
            model=model,
            model_name='Faster R-CNN R50',
            device='cuda'
        )
    """
    adapters = {
        'rtdetr': RTDETRAdapter,
        'yolo': YOLOAdapter,
        'torchvision': TorchvisionAdapter,
    }

    if model_type.lower() not in adapters:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(adapters.keys())}")

    return adapters[model_type.lower()](**kwargs)
