# 100% Free 24/7 Deployment Dockerfile (Hugging Face Spaces / Render / Self-Hosted)
# Hugging Face Spaces Free Tier runs Docker containers on port 7860 with 16 GB RAM & 0 API costs!
FROM python:3.11-slim

WORKDIR /app

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY requirements.txt requirements-llm.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir transformers>=4.46.0 peft>=0.13.0 accelerate>=1.0.0 && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless

COPY --chown=user:user . /app

# Train local ML suite AND fine-tune free open-source Transformer (LoRA) on Harony materials (zero API keys!)
USER user
ENV HAROONY_LOCAL_LLM=true \
    HAROONY_FREE_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
RUN python3 scripts/train_haroony_ai.py && \
    python3 scripts/train_free_llm.py --max-samples 320 --epochs 1

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
