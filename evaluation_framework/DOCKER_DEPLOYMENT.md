# RT-DETR Evaluation - Docker Deployment Guide for A100 Server

Complete guide for running RT-DETR evaluation on your university's A100 server using Docker.

## Prerequisites

- Access to A100 server with Docker installed
- NVIDIA Docker runtime (nvidia-docker2)
- SSH access to the server

## Quick Start (TL;DR)

```bash
# 1. Transfer files to server
scp -r RT-DETR-main your-username@server:/path/to/workspace/

# 2. SSH into server
ssh your-username@server
cd /path/to/workspace/RT-DETR-main/evaluation_framework

# 3. Build and run
docker-compose build
docker-compose run rtdetr-eval

# 4. Inside container, setup and run
./setup_data.sh
./run_evaluation.sh
```

---

## Detailed Instructions

### Step 1: Transfer Files to A100 Server

#### Option A: Using SCP (if you have the files locally)

```bash
# From your local machine
cd "c:\My Files\MonoCab\gITHUB rEPO"
tar -czf rtdetr-eval.tar.gz RT-DETR-main/

# Upload to server
scp rtdetr-eval.tar.gz your-username@your-server.edu:/home/your-username/

# SSH into server
ssh your-username@your-server.edu

# Extract files
cd /home/your-username/
tar -xzf rtdetr-eval.tar.gz
cd RT-DETR-main/evaluation_framework
```

#### Option B: Using Git (if repo is on GitHub)

```bash
# SSH into server
ssh your-username@your-server.edu

# Clone repository
git clone https://github.com/your-username/RT-DETR-main.git
cd RT-DETR-main/evaluation_framework
```

### Step 2: Verify Docker and GPU Access

```bash
# Check Docker is installed
docker --version

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Expected output: Should show your A100 GPU(s)
```

### Step 3: Build Docker Image

```bash
# Navigate to evaluation_framework directory
cd /path/to/RT-DETR-main/evaluation_framework

# Build the Docker image (takes ~5-10 minutes)
docker-compose build

# Or build manually without docker-compose:
docker build -t rtdetr-evaluation:latest .
```

### Step 4: Run Docker Container

#### Option A: Using docker-compose (Recommended)

```bash
# Start container with interactive shell
docker-compose run rtdetr-eval

# You'll be inside the container at /workspace
```

#### Option B: Using docker run

```bash
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/dataset:/workspace/dataset \
  -v $(pwd)/pretrained_weights:/workspace/pretrained_weights \
  -v $(pwd)/outputs:/workspace/outputs \
  -e CUDA_VISIBLE_DEVICES=0 \
  rtdetr-evaluation:latest \
  /bin/bash
```

### Step 5: Download Dataset and Weights (Inside Container)

```bash
# Make script executable
chmod +x setup_data.sh

# Run setup script (downloads ~1.3GB)
./setup_data.sh

# This will download:
# - COCO val2017 dataset (~1GB)
# - COCO annotations (~240MB)
# - RT-DETR pretrained weights (~85MB)
```

**Note:** The download happens once and is persisted via Docker volumes, so you won't need to re-download if you restart the container.

### Step 6: Run Evaluation

```bash
# Make script executable
chmod +x run_evaluation.sh

# Run evaluation
./run_evaluation.sh

# Expected runtime on A100: 2-5 minutes for full COCO val (5000 images)
```

### Step 7: View Results

Results are saved in the `outputs/` directory:

```bash
# Inside container or on host
ls -lh outputs/

# View JSON results
cat outputs/rtdetrv2_r18vd_results.json

# Copy results to your local machine (from host, not container)
# Exit container first (Ctrl+D or exit)
scp -r your-username@server:/path/to/RT-DETR-main/evaluation_framework/outputs ./local-results/
```

---

## Alternative Workflows

### Option 1: Run Jupyter Notebook (Interactive)

```bash
# Start Jupyter service
docker-compose --profile jupyter up rtdetr-jupyter

# Access from your local machine (port forward via SSH)
# On your local machine:
ssh -N -L 8888:localhost:8888 your-username@server

# Then open browser to: http://localhost:8888
# Open RTDETR_Evaluation.ipynb and run cells
```

### Option 2: Run Python Script Directly

```bash
# Inside container
python3 RTDETR_Evaluation_Script.py
```

### Option 3: Test Multiple Models

Modify the configuration in the notebook or script:

```python
# For RT-DETRv2-R50
MODEL_CONFIG = {
    'model_name': 'rtdetrv2_r50vd',
    'config_file': 'configs/rtdetrv2/rtdetrv2_r50vd_120e_coco.yml',
    'checkpoint_path': '../pretrained_weights/rtdetrv2_r50vd_120e_coco.pth',
}

# For RT-DETRv2-R101
MODEL_CONFIG = {
    'model_name': 'rtdetrv2_r101vd',
    'config_file': 'configs/rtdetrv2/rtdetrv2_r101vd_120e_coco.yml',
    'checkpoint_path': '../pretrained_weights/rtdetrv2_r101vd_120e_coco.pth',
}
```

---

## Performance Optimization for A100

### 1. Increase Batch Size

The A100 has much more memory than the RTX 4060. You can increase batch size:

```python
CONFIG = {
    'batch_size': 64,  # or even 128 on A100
    # ... other settings
}
```

### 2. Use Mixed Precision (FP16)

For faster inference:

```python
# Add to evaluation script
model = model.half()  # Convert to FP16
img_tensor = img_tensor.half()  # Convert input to FP16
```

### 3. Compile Model (PyTorch 2.0+)

```python
model = torch.compile(model, mode='reduce-overhead')
```

---

## Expected Performance on A100

Based on the notebook results from RTX 4060:

| Metric | RTX 4060 Laptop | A100 (Expected) |
|--------|-----------------|-----------------|
| **FPS** | 34.10 | 100-150 |
| **Latency** | 29.33ms | 7-10ms |
| **Eval Time** | ~4-5 min | ~1-2 min |
| **Batch Size** | 32 | 128+ |

---

## Troubleshooting

### Issue: GPU not detected

```bash
# Check GPU visibility
nvidia-smi

# Check Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-docker2:
# sudo apt-get install nvidia-docker2
# sudo systemctl restart docker
```

### Issue: Out of memory

```bash
# Reduce batch size in CONFIG
# Or use smaller model (R18 instead of R101)
```

### Issue: Permission denied on scripts

```bash
chmod +x setup_data.sh run_evaluation.sh
```

### Issue: Dataset download fails

```bash
# Download manually and copy to dataset/coco/
# Or check firewall/proxy settings
```

---

## Advanced: Running Multiple Evaluations in Background

```bash
# Start container in detached mode
docker-compose up -d rtdetr-eval

# Execute commands in running container
docker-compose exec rtdetr-eval ./setup_data.sh
docker-compose exec rtdetr-eval ./run_evaluation.sh

# View logs
docker-compose logs -f rtdetr-eval

# Stop container
docker-compose down
```

---

## Cleaning Up

```bash
# Remove container (keeps volumes)
docker-compose down

# Remove container and volumes (deletes downloaded data)
docker-compose down -v

# Remove Docker image
docker rmi rtdetr-evaluation:latest
```

---

## File Structure After Setup

```
RT-DETR-main/evaluation_framework/
├── Dockerfile
├── docker-compose.yml
├── setup_data.sh
├── run_evaluation.sh
├── RTDETR_Evaluation.ipynb
├── configs/
│   └── rtdetrv2/
├── src/
├── dataset/
│   └── coco/
│       ├── val2017/           (5000 images)
│       └── annotations/        (instances_val2017.json)
├── pretrained_weights/
│   ├── rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
│   ├── rtdetrv2_r50vd_120e_coco.pth
│   └── rtdetrv2_r101vd_120e_coco.pth
└── outputs/
    ├── rtdetrv2_r18vd_results.json
    └── rtdetrv2_r18vd_results.png
```

---

## Support

If you encounter issues:

1. Check container logs: `docker-compose logs`
2. Verify GPU access: `nvidia-smi` inside container
3. Check file permissions: `ls -la`
4. Ensure sufficient disk space: `df -h`

---

## Summary Commands

```bash
# Complete workflow
ssh your-username@server
cd /path/to/RT-DETR-main/evaluation_framework
docker-compose build
docker-compose run rtdetr-eval
./setup_data.sh
./run_evaluation.sh
# Results in outputs/ directory
```

Good luck with your evaluation! 🚀
