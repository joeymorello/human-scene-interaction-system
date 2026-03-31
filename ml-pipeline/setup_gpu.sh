#!/usr/bin/env bash
# GPU environment setup for HSI ML Pipeline
# Run this on the RunPod GPU instance
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"
MODELS_DIR="$SCRIPT_DIR/../models"

echo "=== HSI ML Pipeline GPU Setup ==="

# 1. Clone JOSH
if [ ! -d "$VENDOR_DIR/JOSH" ]; then
    echo "[1/4] Cloning JOSH (Joint Optimization for 4D Human-Scene Reconstruction)..."
    git clone https://github.com/genforce/JOSH.git "$VENDOR_DIR/JOSH"
else
    echo "[1/4] JOSH already cloned."
fi

# 2. Clone SAM3
if [ ! -d "$VENDOR_DIR/sam3" ]; then
    echo "[2/4] Cloning SAM3 (Segment Anything Model 3)..."
    git clone https://github.com/facebookresearch/sam3.git "$VENDOR_DIR/sam3"
else
    echo "[2/4] SAM3 already cloned."
fi

# 3. Install JOSH dependencies
echo "[3/4] Installing JOSH dependencies..."
cd "$VENDOR_DIR/JOSH"
pip install -e . 2>/dev/null || pip install -r requirements.txt 2>/dev/null || echo "  Warning: Install JOSH deps manually if needed."
cd "$SCRIPT_DIR"

# 4. Install SAM3 dependencies
echo "[4/4] Installing SAM3 dependencies..."
cd "$VENDOR_DIR/sam3"
pip install -e . 2>/dev/null || pip install -r requirements.txt 2>/dev/null || echo "  Warning: Install SAM3 deps manually if needed."
cd "$SCRIPT_DIR"

# 5. Verify SMPL-X model
if [ -f "$MODELS_DIR/smplx/SMPLX_NEUTRAL.npz" ]; then
    echo ""
    echo "✓ SMPL-X model found at $MODELS_DIR/smplx/SMPLX_NEUTRAL.npz"
else
    echo ""
    echo "⚠ SMPL-X model not found. Copy it to: $MODELS_DIR/smplx/SMPLX_NEUTRAL.npz"
    echo "  Or register at https://smpl-x.is.tue.mpg.de to download."
fi

echo ""
echo "=== Setup complete ==="
echo "Next: pip install -e . && python scripts/run_pipeline.py --help"
