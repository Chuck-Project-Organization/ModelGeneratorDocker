FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git libgl1-mesa-glx libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
