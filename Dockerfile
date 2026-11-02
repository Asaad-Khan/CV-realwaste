FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/baseline.pt \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

COPY requirements-inference.txt .
RUN python -m pip install --no-cache-dir --timeout 120 --retries 10 \
    -r requirements-inference.txt

RUN useradd --create-home --uid 10001 appuser

COPY src/ ./src/
COPY app/ ./app/
COPY models/baseline.pt ./models/baseline.pt

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]