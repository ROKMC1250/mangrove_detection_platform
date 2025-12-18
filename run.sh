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
    export EE_SERVICE_ACCOUNT="$(jq -r .client_email /home/hjh1037/Mangrove_segmentation/mangrove_platform/new_platform/backend/ee-service-account-key.json)"
    export EE_SERVICE_ACCOUNT_KEY="/home/hjh1037/Mangrove_segmentation/mangrove_platform/new_platform/backend/ee-service-account-key.json"
    export GOOGLE_APPLICATION_CREDENTIALS="$EE_SERVICE_ACCOUNT_KEY"
    export GCS_BUCKET="mangrove-gee-exports-2025"
    export GCS_BUCKET_REGION="asia-northeast3"
	export MODEL1_LOG_DIR=/home/hjh1037/Mangrove_segmentation/logs/segformer_mit_b2_v1
	export MODEL_ROOT=/home/hjh1037/Mangrove_segmentation

cd /home/hjh1037/Mangrove_segmentation/mangrove_platform/new_platform

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