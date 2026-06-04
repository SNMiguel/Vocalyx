FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch — avoids 2GB+ GPU builds; all models run fine on CPU
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchaudio==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/embeddings

ENV PYTHONPATH=/app
# Point HuggingFace cache at the Railway persistent volume so models survive deploys
ENV HF_HOME=/app/data/hf_cache

EXPOSE 8000

CMD python -m uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
