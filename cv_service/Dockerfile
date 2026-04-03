FROM python:3.11-slim

# System deps (OpenCV headless needs libgl)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8 weights at build time
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

COPY . .

CMD ["python", "main.py"]
