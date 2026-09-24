#!/usr/bin/env bash
# Bring up the full stack locally: infra containers, migrations, seed data,
# RAG ingest, then the API in the foreground. Run the UI in a second terminal
# (printed at the end) -- Streamlit needs its own terminal/process anyway.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -f .env ]; then
  echo "No .env found -- copying .env.example. Edit it if you want a real LLM provider."
  cp .env.example .env
fi

echo "==> 1/6 Starting Postgres, Qdrant, Redis (docker compose)"
docker compose -f infra/docker-compose.yml up -d postgres qdrant redis

echo "==> 2/6 Waiting for Postgres to accept connections"
until docker compose -f infra/docker-compose.yml exec -T postgres pg_isready -U agent_runtime >/dev/null 2>&1; do
  sleep 1
done

echo "==> 3/6 Waiting for Qdrant to accept connections"
until curl -sf http://localhost:6333/collections >/dev/null 2>&1; do
  sleep 1
done

echo "==> 4/6 Running database migrations"
alembic upgrade head

echo "==> 5/6 Seeding demo data + ingesting sample RAG corpus"
python -m db.seed
python -m rag.ingest data/sample_docs

echo "==> 6/6 Starting the API on http://localhost:8000"
echo
echo "In another terminal, run:"
echo "    source .venv/bin/activate && streamlit run ui/app.py"
echo
exec uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
