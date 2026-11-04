
# Build arguments shared by the stages
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown

# ==========================================
# Stage 1: Train the machine learning model
# ==========================================
FROM python:3.10.14-slim-bookworm AS trainer

ARG MODEL_VERSION

WORKDIR /app

# Install training dependencies first
COPY requirements-train.txt .

RUN pip install --no-cache-dir -r requirements-train.txt

# Copy only the files required for training
COPY training/ ./training/
COPY params.yaml .
COPY data/raw/breast_cancer.csv ./data/raw/breast_cancer.csv

# Train the model inside the image
RUN python training/train.py \
    --data-path data/raw/breast_cancer.csv \
    --output-dir artifacts \
    --model-version "${MODEL_VERSION}"


# ==========================================
# Stage 2: Production API image
# ==========================================
FROM python:3.10.14-slim-bookworm AS production

ARG MODEL_VERSION
ARG GIT_SHA

LABEL org.opencontainers.image.title="Breast Cancer Triage Service"
LABEL org.opencontainers.image.version="${MODEL_VERSION}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"

ENV MODEL_VERSION="${MODEL_VERSION}" \
    GIT_SHA="${GIT_SHA}" \
    MODEL_PATH="artifacts/model.joblib" \
    MODEL_CARD_PATH="artifacts/model_card.json" \
    DECISION_THRESHOLD="0.5" \
    LOG_LEVEL="INFO" \
    REDIS_HOST="redis" \
    REDIS_PORT="6379" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1"

WORKDIR /app

# Install serving dependencies only
COPY requirements-serve.txt .

RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy the trained artifacts from the training stage
COPY --from=trainer /app/artifacts/ ./artifacts/

# Copy the API application
COPY app/ ./app/

# Create a non-root user and give it ownership of the application
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Check API health using Python's standard library (no curl required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Fixed server executable; default arguments can be overridden at runtime
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
