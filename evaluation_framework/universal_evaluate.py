"""
Universal Object Detection Evaluation Script
Supports: RT-DETR, YOLO, Torchvision models, and any COCO-format detector

Usage:
    python universal_evaluate.py --model-type rtdetr --config configs/rtdetrv2_r18vd.yml --checkpoint weights/rtdetr.pth
    python universal_evaluate.py --model-type yolo --model-path yolov8n.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from model_adapters import create_model_adapter


def evaluate_coco(adapter, coco_gt, img_folder, image_ids, device='cuda'):
    """
    Run evaluation on COCO dataset

    Args:
        adapter: Model adapter instance
        coco_gt: COCO ground truth API
        img_folder: Path to images
        image_ids: List of image IDs to evaluate
        device: Device to use

    Returns:
        dict: Evaluation metrics
    """
    all_results = []
    inference_times = []

    print(f"\n{'='*80}")
    print("RUNNING INFERENCE")
    print(f"{'='*80}")
    print(f"Processing {len(image_ids)} images...")

    for img_id in tqdm(image_ids, desc="Inference"):
        # Load image
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_folder, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_size = (img_info['height'], img_info['width'])

        # Run inference
        start_time = time.time()
        predictions = adapter.predict(image, orig_size)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Convert to COCO format
        for pred in predictions:
            all_results.append({
                'image_id': int(img_id),
                'category_id': pred['category_id'],
                'bbox': pred['bbox'],
                'score': pred['score']
            })

    # Compute metrics
    print(f"\n{'='*80}")
    print("COMPUTING COCO METRICS")
    print(f"{'='*80}")
    print(f"Total predictions: {len(all_results)}")

    if len(all_results) == 0:
        print("ERROR: No predictions generated!")
        return None

    # Save predictions
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(all_results, f)
        pred_file = f.name

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    metrics = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
        'AR': coco_eval.stats[6],
        'avg_inference_time_ms': np.mean(inference_times) * 1000,
        'fps': 1.0 / np.mean(inference_times) if len(inference_times) > 0 else 0,
    }

    # Clean up
    os.unlink(pred_file)

    return metrics


def benchmark_fps(adapter, coco_gt, img_folder, image_ids, device='cuda', iterations=100, warmup=10):
    """Benchmark FPS on a single image"""
    print(f"\n{'='*80}")
    print("FPS BENCHMARKING")
    print(f"{'='*80}")

    # Use first image
    img_id = image_ids[0]
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = os.path.join(img_folder, img_info['file_name'])
    image = Image.open(img_path).convert('RGB')
    orig_size = (img_info['height'], img_info['width'])

    print(f"Image: {img_info['file_name']} ({img_info['width']}x{img_info['height']})")

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = adapter.predict(image, orig_size)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    for _ in tqdm(range(iterations), desc="FPS Benchmark"):
        start = time.time()
        _ = adapter.predict(image, orig_size)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / mean_time

    print(f"\nPerformance Metrics:")
    print(f"  FPS:          {fps:.2f} frames/second")
    print(f"  Latency:      {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
    print(f"  Min latency:  {min(times) * 1000:.2f} ms")
    print(f"  Max latency:  {max(times) * 1000:.2f} ms")

    return {
        'fps': fps,
        'latency_ms': mean_time * 1000,
        'std_ms': std_time * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description='Universal Object Detection Evaluation')

    # Model configuration
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['rtdetr', 'yolo', 'torchvision'],
                        help='Type of model to evaluate')

    # RT-DETR specific
    parser.add_argument('--config', type=str, help='Path to RT-DETR config file')
    parser.add_argument('--checkpoint', type=str, help='Path to RT-DETR checkpoint')

    # YOLO specific
    parser.add_argument('--model-path', type=str, help='Path to YOLO model file')

    # Dataset
    parser.add_argument('--img-folder', type=str, default='dataset/coco/val2017',
                        help='Path to COCO images')
    parser.add_argument('--ann-file', type=str, default='dataset/coco/annotations/instances_val2017.json',
                        help='Path to COCO annotations')

    # Evaluation settings
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Number of images to evaluate (default: all)')
    parser.add_argument('--skip-fps', action='store_true',
                        help='Skip FPS benchmarking')

    # Output
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    # Validate arguments
    if args.model_type == 'rtdetr' and (not args.config or not args.checkpoint):
        parser.error("--config and --checkpoint required for RT-DETR")
    if args.model_type == 'yolo' and not args.model_path:
        parser.error("--model-path required for YOLO")

    print(f"\n{'='*80}")
    print("UNIVERSAL OBJECT DETECTION EVALUATION")
    print(f"{'='*80}")

    # Create model adapter
    print(f"\nLoading {args.model_type.upper()} model...")
    if args.model_type == 'rtdetr':
        adapter = create_model_adapter(
            'rtdetr',
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    elif args.model_type == 'yolo':
        adapter = create_model_adapter(
            'yolo',
            model_path=args.model_path,
            device=args.device
        )
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented in CLI")

    # Print model info
    model_info = adapter.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # Load COCO dataset
    print(f"\nLoading COCO dataset...")
    print(f"  Images: {args.img_folder}")
    print(f"  Annotations: {args.ann_file}")
    coco_gt = COCO(args.ann_file)
    image_ids = coco_gt.getImgIds()

    if args.num_images:
        image_ids = image_ids[:args.num_images]
        print(f"  Evaluating on {len(image_ids)} images (subset)")
    else:
        print(f"  Evaluating on {len(image_ids)} images (full dataset)")

    # Run evaluation
    metrics = evaluate_coco(adapter, coco_gt, args.img_folder, image_ids, args.device)

    if metrics is None:
        print("Evaluation failed!")
        return

    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"AP (IoU=0.50:0.95): {metrics['AP']:.1%}")
    print(f"AP50 (IoU=0.50):    {metrics['AP50']:.1%}")
    print(f"AP75 (IoU=0.75):    {metrics['AP75']:.1%}")
    print(f"AP (small):         {metrics['AP_small']:.1%}")
    print(f"AP (medium):        {metrics['AP_medium']:.1%}")
    print(f"AP (large):         {metrics['AP_large']:.1%}")
    print(f"\nInference Speed:")
    print(f"  FPS: {metrics['fps']:.2f}")
    print(f"  Avg time: {metrics['avg_inference_time_ms']:.2f}ms")

    # FPS Benchmark
    fps_metrics = None
    if not args.skip_fps:
        fps_metrics = benchmark_fps(adapter, coco_gt, args.img_folder, image_ids, args.device)

    # Save results
    results = {
        'model_info': model_info,
        'evaluation': metrics,
        'fps_benchmark': fps_metrics,
        'dataset': {
            'num_images': len(image_ids),
            'img_folder': args.img_folder,
            'ann_file': args.ann_file,
        }
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
