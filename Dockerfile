FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Fix apt mirror issues by switching to HTTPS
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-pip \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
