FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    unzip \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers \
    torchvision \
    scipy \
    tqdm \
    pytest \
    vastai \
    kaggle

# Copy repository
COPY . /workspace/Adaptive-Precision-Ascension

WORKDIR /workspace/Adaptive-Precision-Ascension

# Build native C++ CUDA extension inplace
RUN pip install -e ".[examples]" || pip install -e .

CMD ["python", "scripts/run_vastai_cluster1.py", "--epochs", "50", "--batch_size", "16", "--cuda_graph", "--on_finish", "stop"]
