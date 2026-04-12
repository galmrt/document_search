#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtualenv if present
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  docker-compose stop kibana elasticsearch
  exit 0
}
trap cleanup SIGINT SIGTERM

# 1. Start Elasticsearch and Kibana
echo "Starting Elasticsearch and Kibana..."
docker-compose up -d elasticsearch kibana

# 2. Wait for Elasticsearch to be healthy
echo "Waiting for Elasticsearch..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:9200/_cluster/health > /dev/null 2>&1; then
    echo "Elasticsearch is ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "Elasticsearch did not become healthy in time. Exiting."
    exit 1
  fi
  sleep 2
done

# 3. Start backend
echo "Starting backend (FastAPI)..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 4. Start frontend
echo "Starting frontend (Streamlit)..."
streamlit run ui.py --server.port 8501 &
FRONTEND_PID=$!

echo ""
echo "All services running:"
echo "  Elasticsearch: http://localhost:9200"
echo "  Kibana:        http://localhost:5601"
echo "  Backend API:   http://localhost:8000"
echo "  Frontend UI:   http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services."

wait "$BACKEND_PID" "$FRONTEND_PID"
