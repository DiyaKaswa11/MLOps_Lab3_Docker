# ---------- Shared base: small Python image ----------
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# ---------- Stage 1: train the model ----------
FROM base AS trainer
# Install packages before copying code, so code changes do not reinstall packages
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt
COPY src/train.py .
ENV ARTIFACT_DIR=/app/artifacts
RUN python train.py
# When run with Docker Compose, retrain using the current environment variables
CMD ["python", "train.py"]

# ---------- Stage 2: serve the model ----------
FROM base AS serving
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
# Run as a regular user instead of root
RUN useradd --create-home --uid 1000 appuser
COPY src/app.py .
COPY src/templates ./templates
# Only the trained model is copied from the training stage, not the training code
COPY --from=trainer /app/artifacts ./artifacts
ENV ARTIFACT_DIR=/app/artifacts
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
# Production server instead of Flask's development server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "app:app"]
