#!/usr/bin/env bash
# Run this AFTER gaining access at https://huggingface.co/facebook/sam3
#
# It will:
#   1. download facebook/sam3 checkpoint via HuggingFace Hub (~3.5 GB)
#   2. start the platform server on port 8000
#   3. tail the server log so you can see SAM3 init messages
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export VIRTUAL_ENV="$PROJECT_ROOT/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

echo "==> 1. Verifying HuggingFace access to facebook/sam3 ..."
"$VIRTUAL_ENV/bin/python" - <<'PY'
import sys
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError
api = HfApi()
me = api.whoami()
print(f"   logged in as: {me.get('name')}")
try:
    hf_hub_download(repo_id="facebook/sam3", filename="config.json")
    print("   access OK")
except GatedRepoError:
    print("   ERROR: still gated. Visit https://huggingface.co/facebook/sam3 and click "
          "\"Agree and access repository\". Re-run this script once granted.")
    sys.exit(1)
PY

echo ""
echo "==> 2. Downloading SAM3 checkpoint (~3.5 GB if first run) ..."
"$VIRTUAL_ENV/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
import os
print(f"   {ckpt} ({os.path.getsize(ckpt) / 1e9:.2f} GB)")
PY

echo ""
echo "==> 3. Starting server on port 8000 ..."

# Kill anything on 8000
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "   port 8000 in use — killing old process"
    kill -9 "$(lsof -ti:8000)" 2>/dev/null || true
    sleep 1
fi

# Project env from run.sh (kept minimal here; run.sh is the canonical setup).
export EE_SERVICE_ACCOUNT_KEY="${EE_SERVICE_ACCOUNT_KEY:-$PROJECT_ROOT/backend/ee-service-account-key.json}"
export GCS_BUCKET="${GCS_BUCKET:-mangrove-gee-exports-2025}"
export GCS_BUCKET_REGION="${GCS_BUCKET_REGION:-asia-northeast3}"

mkdir -p "$PROJECT_ROOT/logs"
LOG="$PROJECT_ROOT/logs/server.log"
nohup "$VIRTUAL_ENV/bin/python" -m uvicorn backend.main:app \
    --host 0.0.0.0 --port 8000 --reload \
    > "$LOG" 2>&1 &
SERVER_PID=$!
echo "   server pid $SERVER_PID, log: $LOG"
echo ""

echo "==> 4. Waiting for SAM3 model to load (this takes ~30-60s on first run) ..."
for i in $(seq 1 60); do
    sleep 2
    status=$(curl -s http://localhost:8000/api/sam3/status 2>/dev/null || true)
    if [ -n "$status" ]; then
        echo "   /api/sam3/status -> $status"
        if echo "$status" | grep -q '"ready":[[:space:]]*true'; then
            echo ""
            echo "==> ✅ SAM3 is ready. Open http://localhost:8000"
            exit 0
        fi
    fi
done

echo ""
echo "==> ⚠️  SAM3 didn't report ready in 2 min. Last status: ${status:-(no response)}"
echo "    Tailing the log so you can see what happened:"
tail -40 "$LOG"
exit 1
