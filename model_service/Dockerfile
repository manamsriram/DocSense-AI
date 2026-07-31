# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download models at build time so cold starts need no internet access.
RUN PYTHONPATH=/install/lib/python3.11/site-packages python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/model-cache'); \
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/model-cache'); \
print('Models pre-downloaded.') \
"

# ---- Stage 2: runtime ----
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /model-cache /model-cache
COPY service.py .

ENV FASTEMBED_CACHE_PATH=/model-cache

EXPOSE 10000

# Single worker process — models live in process memory, this service's sole
# job, and a second worker would double that memory on a 512MB dyno. Threads
# instead of pure sync: batched /rerank calls block long enough (multiple
# ONNX sub-batches) that Render's own /health probe times out on a sync
# worker and kills the instance mid-request; ONNX releases the GIL during
# inference so a health-check thread can still respond while rerank runs.
# max-requests recycles the worker periodically since ONNX's memory arena
# grows across inference calls and doesn't fully release back to the OS.
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", \
     "--worker-class", "gthread", "--threads", "4", "--timeout", "60", \
     "--max-requests", "40", "--max-requests-jitter", "10", "service:app"]
