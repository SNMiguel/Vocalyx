FROM python:3.10-slim

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ARM64 (Oracle Ampere A1): standard PyPI wheels are CPU-only on this arch — no +cpu index needed
# x86_64: also works; pip selects the right wheel automatically
RUN pip install --no-cache-dir torch torchaudio

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/embeddings

ENV PYTHONPATH=/app
# Point HuggingFace cache at the Railway persistent volume so models survive deploys
ENV HF_HOME=/app/data/hf_cache

EXPOSE 8000

CMD python -m uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
