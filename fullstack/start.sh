#!/bin/bash
# start.sh — Launch all three services for the CI/CD Prediction Dashboard
# Run from the project root: bash fullstack/start.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FULLSTACK="$ROOT/fullstack"

echo "🚀 Starting CI/CD Prediction Dashboard..."
echo "   Project root: $ROOT"
echo ""

# 1. Python ML microservice
echo "▶ [1/3] Starting ML Service (FastAPI) on :8001"
cd "$FULLSTACK/ml-service"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload &
ML_PID=$!

# 2. Node.js backend
echo "▶ [2/3] Starting Backend (Express) on :3001"
cd "$FULLSTACK/backend"
node server.js &
BACKEND_PID=$!

# 3. React frontend
echo "▶ [3/3] Starting Frontend (Vite/React) on :5173"
cd "$FULLSTACK/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ All services started!"
echo ""
echo "   🌐 Dashboard  → http://localhost:5173"
echo "   🔌 Backend    → http://localhost:3001"
echo "   🤖 ML Service → http://localhost:8001"
echo ""
echo "   Demo login:   demo@cicd.ai  /  demo1234"
echo ""
echo "Press Ctrl+C to stop all services."

trap "echo ''; echo 'Stopping all services...'; kill $ML_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
