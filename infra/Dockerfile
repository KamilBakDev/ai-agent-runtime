# syntax=docker/dockerfile:1

# --- Builder: install dependencies into a venv ---
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Runtime: slim image with just the venv + source ---
FROM python:3.11-slim AS runtime

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv
COPY agents ./agents
COPY api ./api
COPY rag ./rag
COPY db ./db
COPY ui ./ui
COPY data ./data
COPY alembic.ini ./alembic.ini

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
