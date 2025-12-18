#!/usr/bin/bash
set -euo pipefail

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
	echo "📦 Creating virtual environment..."
	python3 -m venv venv
fi

# Activate venv
# shellcheck disable=SC1091
source venv/bin/activate

export PYTHONUNBUFFERED=1

# Set project root (adjust if needed)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Set Earth Engine credentials (adjust paths if needed)
if [ -z "${EE_SERVICE_ACCOUNT_KEY:-}" ]; then
    export EE_SERVICE_ACCOUNT_KEY="$PROJECT_ROOT/backend/ee-service-account-key.json"
fi

if [ -z "${EE_SERVICE_ACCOUNT:-}" ] && [ -f "$EE_SERVICE_ACCOUNT_KEY" ]; then
    if command -v jq &> /dev/null; then
        export EE_SERVICE_ACCOUNT="$(jq -r .client_email "$EE_SERVICE_ACCOUNT_KEY")"
    fi
fi

if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$EE_SERVICE_ACCOUNT_KEY"
fi

# Set GCS bucket (can be overridden by environment variables)
if [ -z "${GCS_BUCKET:-}" ]; then
    export GCS_BUCKET="mangrove-gee-exports-2025"
fi

if [ -z "${GCS_BUCKET_REGION:-}" ]; then
    export GCS_BUCKET_REGION="asia-northeast3"
fi

# Model paths - set these environment variables if you have models
# If not set, the segmentation model will be disabled but other features will work
# export MODEL1_LOG_DIR=/path/to/your/model/logs
# export MODEL_ROOT=/path/to/your/model/root

python -m pip install -r backend/requirements.txt

# Get IP addresses
echo "======================================"
echo "🚀 Starting Mangrove Platform Server"
echo "======================================"
echo ""
echo "📍 Server will be accessible at:"
echo "   - Local:   http://localhost:8000"
echo "   - Local:   http://127.0.0.1:8000"

# Get all network interfaces
if command -v hostname &> /dev/null; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    if [ -n "$LOCAL_IP" ]; then
        echo "   - Network: http://$LOCAL_IP:8000"
    fi
fi

# Alternative method to get IP
if command -v ip &> /dev/null; then
    IP_ADDR=$(ip route get 1 2>/dev/null | grep -Po '(?<=src )[\d.]+' || echo "")
    if [ -n "$IP_ADDR" ] && [ "$IP_ADDR" != "$LOCAL_IP" ]; then
        echo "   - Network: http://$IP_ADDR:8000"
    fi
fi

echo ""
echo "📂 Frontend is served from /static/"
echo "======================================"
echo ""

exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload 