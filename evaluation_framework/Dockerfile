# RT-DETR Evaluation Docker Image
# Optimized for NVIDIA A100 GPU

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
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

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
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

# Create working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create directories for data and outputs
RUN mkdir -p /workspace/dataset/coco \
    /workspace/pretrained_weights \
    /workspace/outputs \
    /workspace/outputs_universal

# Set permissions
RUN chmod +x /workspace/*.sh 2>/dev/null || true

# Expose Jupyter port (optional)
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
