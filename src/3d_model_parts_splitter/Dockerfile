FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# system deps: GPU-related libs you already had + build toolchain for native wheels
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git \
      libgl1-mesa-glx libglib2.0-0 libx11-6 libxrender1 libgl1 \
      build-essential cmake python3-dev pkg-config \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch CUDA 12.4
RUN pip install --no-cache-dir \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# torch-scatter matching torch 2.4.0 + cu124
RUN pip install --no-cache-dir \
  torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

COPY PartField /workspace/PartField
ENV PF_ROOT=/workspace/PartField/partfield
ENV PF_CKPT=/runpod-volume/3d_model_parts_splitter/model_objaverse.ckpt

COPY handler.py /workspace/handler.py
CMD ["python3", "-u", "handler.py"]
