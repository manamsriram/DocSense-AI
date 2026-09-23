# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN python -m spacy download en_core_web_sm --target=/install/lib/python3.11/site-packages

# ---- Stage 2: runtime ----
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

RUN mkdir -p pdfFolder

EXPOSE 10000

# Single worker — ML models and BM25 index live in process memory
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120", "app:app"]
