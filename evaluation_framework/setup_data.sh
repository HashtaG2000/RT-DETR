#!/bin/bash

# RT-DETR Evaluation Setup Script
# Downloads datasets and pretrained weights

set -e  # Exit on error

echo "=============================================="
echo "RT-DETR Evaluation Setup"
echo "=============================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running inside Docker
if [ -f /.dockerenv ]; then
    print_info "Running inside Docker container"
    WORKSPACE="/workspace"
else
    print_info "Running on host system"
    WORKSPACE="."
fi

cd "$WORKSPACE"

# ============================================
# 1. Download COCO Dataset
# ============================================

print_info "Step 1/2: Setting up COCO dataset"

DATASET_DIR="$WORKSPACE/dataset/coco"
mkdir -p "$DATASET_DIR"

# Download validation images
if [ ! -d "$DATASET_DIR/val2017" ]; then
    print_info "Downloading COCO val2017 images (~1GB)..."
    cd "$DATASET_DIR"
    wget -c http://images.cocodataset.org/zips/val2017.zip
    print_info "Extracting images..."
    unzip -q val2017.zip
    rm val2017.zip
    print_info "COCO images downloaded successfully"
else
    print_info "COCO val2017 images already exist, skipping download"
fi

# Download annotations
if [ ! -f "$DATASET_DIR/annotations/instances_val2017.json" ]; then
    print_info "Downloading COCO annotations (~241MB)..."
    cd "$DATASET_DIR"
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    print_info "Extracting annotations..."
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    print_info "COCO annotations downloaded successfully"
else
    print_info "COCO annotations already exist, skipping download"
fi

# Verify dataset
VAL_IMAGES=$(ls "$DATASET_DIR/val2017" 2>/dev/null | wc -l)
print_info "COCO val2017 dataset ready: $VAL_IMAGES images"

# ============================================
# 2. Download Pretrained Weights
# ============================================

print_info "Step 2/2: Downloading pretrained weights"

WEIGHTS_DIR="$WORKSPACE/pretrained_weights"
mkdir -p "$WEIGHTS_DIR"

cd "$WEIGHTS_DIR"

# RT-DETRv2 R18
if [ ! -f "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth" ]; then
    print_info "Downloading RT-DETRv2-R18 weights (~85MB)..."
    wget -c https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
    print_info "RT-DETRv2-R18 downloaded successfully"
else
    print_info "RT-DETRv2-R18 weights already exist"
fi

# RT-DETRv2 R50 (optional)
if [ ! -f "rtdetrv2_r50vd_120e_coco.pth" ]; then
    print_warn "Downloading RT-DETRv2-R50 weights (optional, ~140MB)..."
    wget -c https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r50vd_120e_coco.pth || print_warn "Failed to download R50, skipping"
fi

# RT-DETRv2 R101 (optional)
if [ ! -f "rtdetrv2_r101vd_120e_coco.pth" ]; then
    print_warn "Downloading RT-DETRv2-R101 weights (optional, ~200MB)..."
    wget -c https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r101vd_120e_coco.pth || print_warn "Failed to download R101, skipping"
fi

# List downloaded weights
print_info "Available model weights:"
ls -lh "$WEIGHTS_DIR"/*.pth 2>/dev/null || print_warn "No weights found"

# ============================================
# 3. Verify Setup
# ============================================

print_info "Verifying setup..."

# Check directories
if [ -d "$DATASET_DIR/val2017" ] && [ -f "$DATASET_DIR/annotations/instances_val2017.json" ]; then
    print_info "✓ COCO dataset ready"
else
    print_error "✗ COCO dataset incomplete"
    exit 1
fi

if [ -f "$WEIGHTS_DIR/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth" ]; then
    print_info "✓ Model weights ready"
else
    print_error "✗ Model weights missing"
    exit 1
fi

# Check GPU (if nvidia-smi available)
if command -v nvidia-smi &> /dev/null; then
    print_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    print_warn "nvidia-smi not found, skipping GPU check"
fi

# ============================================
# Setup Complete
# ============================================

echo ""
echo "=============================================="
print_info "Setup completed successfully!"
echo "=============================================="
echo ""
print_info "Dataset location: $DATASET_DIR"
print_info "Weights location: $WEIGHTS_DIR"
echo ""
print_info "Next steps:"
echo "  1. Run evaluation: ./run_evaluation.sh"
echo "  2. Or start Jupyter: jupyter notebook"
echo "  3. Or run Python script: python3 RTDETR_Evaluation_Script.py"
echo ""
