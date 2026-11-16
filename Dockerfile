# 100% Free 24/7 Deployment Dockerfile (Hugging Face Spaces / Render / Self-Hosted)
# Hugging Face Spaces Free Tier runs Docker containers on port 7860 with 16 GB RAM & 0 API costs!
FROM python:3.11-slim

WORKDIR /app

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless

COPY --chown=user:user . /app

# Ensure models are freshly trained/verified at build time
USER user
RUN python3 scripts/train_haroony_ai.py

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
