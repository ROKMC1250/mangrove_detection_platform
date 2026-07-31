#!/usr/bin/bash
set -euo pipefail

# Set project root (adjust if needed)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1

# Kill any existing uvicorn on port 8000
if lsof -ti:8000 &>/dev/null; then
    echo "⚠️  Port 8000 in use — killing existing process..."
    kill -9 $(lsof -ti:8000) 2>/dev/null
    sleep 1
fi

# --- Python environment setup via uv ---
# SAM3 requires Python 3.12+
PYTHON_VERSION="3.12"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv if it doesn't exist (uv auto-installs Python if needed)
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment (Python ${PYTHON_VERSION})..."
    uv venv --python "$PYTHON_VERSION" venv
fi

# Activate venv — use explicit VIRTUAL_ENV to avoid stale activate scripts
export VIRTUAL_ENV="$PROJECT_ROOT/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME

# --- User configuration ---
# All deployment-specific settings live in .env — copy .env.example to .env and
# edit that one file. Variables already exported in your shell take precedence.
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Earth Engine credentials
if [ -z "${EE_SERVICE_ACCOUNT_KEY:-}" ]; then
    export EE_SERVICE_ACCOUNT_KEY="$PROJECT_ROOT/backend/ee-service-account-key.json"
fi

# Resolve a relative key path against the project root
case "$EE_SERVICE_ACCOUNT_KEY" in
    /*) ;;
    *) export EE_SERVICE_ACCOUNT_KEY="$PROJECT_ROOT/$EE_SERVICE_ACCOUNT_KEY" ;;
esac

# The service account address is read from the key file when not set explicitly
if [ -z "${EE_SERVICE_ACCOUNT:-}" ] && [ -f "$EE_SERVICE_ACCOUNT_KEY" ]; then
    if command -v jq &> /dev/null; then
        export EE_SERVICE_ACCOUNT="$(jq -r .client_email "$EE_SERVICE_ACCOUNT_KEY")"
    fi
fi

if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$EE_SERVICE_ACCOUNT_KEY"
fi

# GCS is optional — set GCS_BUCKET / GCS_BUCKET_REGION in .env to enable
# server-side GeoTIFF export for large AOIs.

# Model paths - set these environment variables if you have models
# If not set, the segmentation model will be disabled but other features will work
# export MODEL1_LOG_DIR=/path/to/your/model/logs
# export MODEL_ROOT=/path/to/your/model/root

# Install PyTorch from the CUDA 12.8 wheel index (required by SAM3) before
# the rest of the deps so it isn't downgraded to a CPU-only build.
uv pip install --python "$VIRTUAL_ENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    "torch>=2.7,<3" "torchvision>=0.22,<1"

uv pip install --python "$VIRTUAL_ENV/bin/python" -r backend/requirements.txt

# Install SAM3 from source if it isn't already available.
# Checkpoints require HuggingFace authentication: run `hf auth login` once.
if ! "$VIRTUAL_ENV/bin/python" -c "import sam3" >/dev/null 2>&1; then
    echo "📦 Installing SAM3 from facebookresearch/sam3..."
    uv pip install --python "$VIRTUAL_ENV/bin/python" \
        "git+https://github.com/facebookresearch/sam3.git"
fi

# Default SAM3 checkpoint directory (can be overridden by env)
if [ -z "${SAM3_CHECKPOINT_DIR:-}" ]; then
    export SAM3_CHECKPOINT_DIR="$PROJECT_ROOT/repo/sam3"
fi

# Get IP addresses
echo "======================================"
echo "🚀 Starting EarthScope Server"
echo "======================================"
echo ""
echo "📍 Server will be accessible at:"
echo "   - Local:   http://localhost:8000"
echo "   - Local:   http://127.0.0.1:8000"

# Function to get local network IPs excluding VPN interfaces
get_local_ips() {
    local ips_seen=""
    
    # Method 1: Get IPs from physical interfaces (exclude VPN interfaces: wg*, tun*, vpn*)
    if command -v ip &> /dev/null; then
        # Get all network interfaces, exclude VPN interfaces
        for iface in $(ip link show | grep -E '^[0-9]+:' | awk -F': ' '{print $2}' | grep -vE '^(lo|wg|tun|vpn)' | head -10); do
            ip_addr=$(ip -4 addr show "$iface" 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1)
            if [ -n "$ip_addr" ] && [ "$ip_addr" != "127.0.0.1" ]; then
                # Avoid duplicates
                if [[ ! " $ips_seen " =~ " $ip_addr " ]]; then
                    echo "   - Network: http://$ip_addr:8000 ($iface)"
                    ips_seen="$ips_seen $ip_addr"
                fi
            fi
        done
    fi
    
    # Method 2: Fallback - get IP from default route (excluding VPN)
    if [ -z "$ips_seen" ] && command -v ip &> /dev/null; then
        # Try to get IP from default route, but check if it's not on a VPN interface
        default_route=$(ip route get 8.8.8.8 2>/dev/null)
        if [ -n "$default_route" ]; then
            default_ip=$(echo "$default_route" | grep -oP 'src \K[\d.]+' || echo "")
            default_iface=$(echo "$default_route" | grep -oP 'dev \K\S+' || echo "")
            
            # Only use if it's not a VPN interface
            if [ -n "$default_ip" ] && [ -n "$default_iface" ] && [[ ! "$default_iface" =~ ^(wg|tun|vpn) ]]; then
                if [[ ! " $ips_seen " =~ " $default_ip " ]]; then
                    echo "   - Network: http://$default_ip:8000 ($default_iface)"
                    ips_seen="$ips_seen $default_ip"
                fi
            fi
        fi
    fi
    
    # Method 3: Last resort - use hostname -I but try to filter VPN IPs
    if [ -z "$ips_seen" ] && command -v hostname &> /dev/null; then
        for ip in $(hostname -I 2>/dev/null); do
            if [[ ! "$ip" =~ ^127\. ]]; then
                # Check which interface this IP belongs to
                iface=$(ip -4 addr | grep "$ip" | awk '{print $NF}' | head -1)
                if [ -n "$iface" ] && [[ ! "$iface" =~ ^(wg|tun|vpn|lo) ]]; then
                    if [[ ! " $ips_seen " =~ " $ip " ]]; then
                        echo "   - Network: http://$ip:8000 ($iface)"
                        ips_seen="$ips_seen $ip"
                    fi
                fi
            fi
        done
    fi
}

# Display local network IPs
get_local_ips

# Check and fix local network routing when VPN is active
fix_local_network_routing() {
    # Check if WireGuard/VPN interface is active
    local vpn_iface=""
    if command -v ip &> /dev/null; then
        vpn_iface=$(ip link show type wireguard 2>/dev/null | head -1 | awk -F': ' '{print $2}' | awk '{print $1}')
    fi
    
    if [ -z "$vpn_iface" ]; then
        # Check for common VPN interface names
        for iface in $(ip link show | grep -E '^[0-9]+:' | awk -F': ' '{print $2}' | grep -E '^(wg|tun|vpn|myvpn)'); do
            if ip link show "$iface" 2>/dev/null | grep -q "state UP"; then
                vpn_iface="$iface"
                break
            fi
        done
    fi
    
    if [ -n "$vpn_iface" ]; then
        echo ""
        echo "🔒 VPN detected: $vpn_iface"
        
        # Get local network routes from physical interfaces
        local routes_fixed=0
        
        # Check each physical interface
        for iface in $(ip link show | grep -E '^[0-9]+:' | awk -F': ' '{print $2}' | grep -vE '^(lo|wg|tun|vpn|myvpn|docker)'); do
            # Get IP and subnet for this interface
            ip_info=$(ip -4 addr show "$iface" 2>/dev/null | grep 'inet ' | awk '{print $2}' | head -1)
            
            if [ -n "$ip_info" ]; then
                ip_addr=$(echo "$ip_info" | cut -d'/' -f1)
                subnet=$(echo "$ip_info" | cut -d'/' -f2)
                
                if [ -n "$subnet" ] && [ "$subnet" != "$ip_info" ]; then
                    # Calculate network address (simple calculation for common subnets)
                    IFS='.' read -r i1 i2 i3 i4 <<< "$ip_addr"
                    case "$subnet" in
                        24) net_addr="$i1.$i2.$i3.0/24" ;;
                        16) net_addr="$i1.$i2.0.0/16" ;;
                        8)  net_addr="$i1.0.0.0/8" ;;
                        *)  net_addr="$i1.$i2.$i3.0/$subnet" ;;
                    esac
                    
                    # Check if route exists and goes through correct interface
                    existing_route=$(ip route show "$net_addr" 2>/dev/null | grep "dev $iface")
                    
                    if [ -z "$existing_route" ]; then
                        # Check if traffic to local network would go through VPN
                        test_ip="$i1.$i2.$i3.1"
                        route_info=$(ip route get "$test_ip" 2>/dev/null)
                        route_dev=$(echo "$route_info" | grep -oP 'dev \K\S+' | head -1 || echo "")
                        
                        if [ "$route_dev" = "$vpn_iface" ]; then
                            echo "⚠️  Local network $net_addr ($iface) routed through VPN"
                            echo "   Adding explicit route..."
                            
                            # Try to add route (with or without sudo)
                            if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
                                # Sudo available and passwordless
                                if sudo ip route add "$net_addr" dev "$iface" 2>/dev/null; then
                                    echo "✅ Fixed: $net_addr via $iface"
                                    routes_fixed=$((routes_fixed + 1))
                                fi
                            elif [ "$EUID" -eq 0 ]; then
                                # Running as root
                                if ip route add "$net_addr" dev "$iface" 2>/dev/null; then
                                    echo "✅ Fixed: $net_addr via $iface"
                                    routes_fixed=$((routes_fixed + 1))
                                fi
                            else
                                echo "⚠️  Need sudo to fix routing. Run manually:"
                                echo "   sudo ip route add $net_addr dev $iface"
                            fi
                        fi
                    fi
                fi
            fi
        done
        
        if [ $routes_fixed -eq 0 ]; then
            echo "✅ Local network routing appears correct"
            echo "ℹ️  If local access still fails, configure WireGuard to exclude local networks:"
            echo "   Edit WireGuard config and modify AllowedIPs to exclude local subnets"
        fi
    fi
}

# Fix local network routing if VPN is active
fix_local_network_routing

echo ""
echo "📂 Frontend is served from /static/"
echo "======================================"
echo ""

exec "$VIRTUAL_ENV/bin/python" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --no-access-log 