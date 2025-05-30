FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip git libgl1-mesa-glx libglib2.0-0 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# RunPod looks for `handler.py`
CMD ["python3", "-u", "handler.py"]
