# Argos prediction API container.
#
# Build:  docker build -t argos-api:dev .
# Run:    docker compose up -d api
#
# Prerequisite: models/ must contain the trained artifacts. Run
#   python -m src.ingest && python -m src.features && python -m src.train
# first so the COPY below has something to find.

FROM python:3.13-slim

# Stop Python from buffering stdout/stderr so `docker logs` is live.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps. Pin torch to the CPU-only wheel before reading
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# Application code, demo UI, and trained artifacts.
COPY src/ ./src/
COPY web/ ./web/
COPY models/fraud_detector_v1.pt \
     models/scaler.pkl \
     models/feature_columns.json \
     ./models/

# Drop privileges — never run a public-facing process as root.
RUN useradd --create-home --shell /bin/bash argos \
 && chown -R argos:argos /app
USER argos

EXPOSE 8000

# A tiny urllib health probe — saves ~5MB vs installing curl.
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=4 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
