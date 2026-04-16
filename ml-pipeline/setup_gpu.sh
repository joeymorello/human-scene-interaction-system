#!/usr/bin/env bash
# GPU environment setup for the HSI ML Pipeline.
#
# Run this on your GPU box (RunPod, etc.). It:
#   1. Clones JOSH (with its submodules) into ml-pipeline/vendor/JOSH
#   2. Clones SAM3 into ml-pipeline/vendor/sam3 (for the downstream segmentor)
#   3. Creates a Python 3.10 venv at .venv-ml (JOSH requires 3.10 for chumpy)
#   4. Installs torch+cu128 and the JOSH requirements
#   5. Applies the JOSH README's parse_chunks patch to TRAM
#   6. Downloads the checkpoints that have public URLs (VIMO, DECO, JOSH3R)
#   7. Prints instructions for the pieces that require a manual login (SMPL)
#
# Target: Ubuntu 22.04 + CUDA 12.8 + ≥24GB VRAM (per JOSH README).
#
# Env overrides:
#   PYTHON_BIN       - python 3.10 interpreter to use (default: python3.10)
#   SKIP_DOWNLOADS   - set to 1 to skip the checkpoint downloads
#   SKIP_PIP         - set to 1 to skip dependency installation
#   JOSH_COMMIT      - pin JOSH to a specific commit/tag (default: main)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"
MODELS_DIR="$REPO_ROOT/models"
VENV_ML="$REPO_ROOT/.venv-ml"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
JOSH_COMMIT="${JOSH_COMMIT:-main}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }

mkdir -p "$VENDOR_DIR" "$MODELS_DIR/smplx"

# ---------------------------------------------------------------- 1. Clone JOSH
log "[1/7] Clone JOSH (with submodules)"
if [ ! -d "$VENDOR_DIR/JOSH/.git" ]; then
    git clone --recursive https://github.com/genforce/JOSH.git "$VENDOR_DIR/JOSH"
else
    echo "  already present — fetching latest"
    git -C "$VENDOR_DIR/JOSH" fetch --recurse-submodules origin
fi
git -C "$VENDOR_DIR/JOSH" checkout "$JOSH_COMMIT"
git -C "$VENDOR_DIR/JOSH" submodule update --init --recursive

# ---------------------------------------------------------------- 2. Clone SAM3
log "[2/7] Clone SAM3"
if [ ! -d "$VENDOR_DIR/sam3/.git" ]; then
    git clone https://github.com/facebookresearch/sam3.git "$VENDOR_DIR/sam3"
else
    echo "  already present"
fi

# ----------------------------------------------------------- 3. Python 3.10 venv
log "[3/7] Create .venv-ml (Python 3.10)"
if [ ! -x "$VENV_ML/bin/python" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.10 "$VENV_ML"
    elif command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        "$PYTHON_BIN" -m venv "$VENV_ML"
    else
        echo "ERROR: need either 'uv' or '$PYTHON_BIN' on PATH."
        echo "  Install uv:  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
else
    echo "  already present at $VENV_ML"
fi

VENV_PIP="$VENV_ML/bin/pip"
VENV_PY="$VENV_ML/bin/python"

"$VENV_PIP" install --upgrade pip wheel setuptools >/dev/null

# ------------------------------------------------------------- 4. Install deps
if [ "${SKIP_PIP:-0}" != "1" ]; then
    log "[4/7] Install torch (cu128) + JOSH requirements"
    "$VENV_PIP" install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128

    (
        cd "$VENDOR_DIR/JOSH"
        "$VENV_PIP" install -r requirements.txt
        "$VENV_PIP" install --no-build-isolation git+https://github.com/mattloper/chumpy
        "$VENV_PIP" install -e .
    )

    # Extras needed by the HSI pipeline itself (not in JOSH's requirements)
    "$VENV_PIP" install fastapi "uvicorn[standard]" python-multipart aiofiles pydantic \
        trimesh scipy smplx rerun-sdk open-clip-torch transformers opencv-python pyyaml
else
    log "[4/7] SKIP_PIP=1 set — skipping dep install"
fi

# --------------------------------------------------------- 5. parse_chunks patch
log "[5/7] Apply TRAM parse_chunks patch (per JOSH README)"
TRAM_TOOLS="$VENDOR_DIR/JOSH/third_party/tram/lib/pipeline/tools.py"
TRAM_VIMO="$VENDOR_DIR/JOSH/third_party/tram/lib/models/hmr_vimo.py"
if [ -f "$TRAM_TOOLS" ] && [ -f "$TRAM_VIMO" ]; then
    "$VENV_PY" - "$TRAM_TOOLS" "$TRAM_VIMO" <<'PY'
import re, sys, pathlib
tools_path = pathlib.Path(sys.argv[1])
vimo_path  = pathlib.Path(sys.argv[2])

tools_src = tools_path.read_text()
vimo_src  = vimo_path.read_text()

if "def parse_chunks" not in vimo_src:
    m = re.search(r"^def parse_chunks\b[\s\S]*?(?=^def |\Z)", tools_src, re.M)
    if not m:
        print("  no parse_chunks found in tools.py — skipping")
        sys.exit(0)
    func_src = m.group(0).rstrip() + "\n\n"
    vimo_path.write_text(func_src + vimo_src)
    tools_path.write_text(tools_src.replace(m.group(0), ""))
    print("  moved parse_chunks -> hmr_vimo.py")
else:
    print("  parse_chunks already in hmr_vimo.py")
PY
else
    warn "TRAM paths not found; skipping parse_chunks move. Looked at $TRAM_TOOLS"
fi

# --------------------------------------------------------- 6. Download checkpts
if [ "${SKIP_DOWNLOADS:-0}" != "1" ]; then
    log "[6/7] Download public checkpoints"
    CKPT_DIR="$VENDOR_DIR/JOSH/data/checkpoints"
    JOSH3R_CKPT_DIR="$REPO_ROOT/models/josh3r"
    mkdir -p "$CKPT_DIR" "$JOSH3R_CKPT_DIR"

    # Ensure gdown for Google Drive files
    "$VENV_PIP" install -q gdown

    # VIMO (HMR) — Google Drive
    if [ ! -f "$CKPT_DIR/vimo_checkpoint.pth.tar" ]; then
        echo "  fetching VIMO..."
        "$VENV_PY" -m gdown \
            "https://drive.google.com/uc?id=1fdeUxn_hK4ERGFwuksFpV_-_PHZJuoiW" \
            -O "$CKPT_DIR/vimo_checkpoint.pth.tar" \
            || warn "VIMO download failed — grab manually from the link in JOSH/README."
    fi

    # DECO (contact — not strictly needed for JOSH3R, but base-JOSH path uses it)
    if [ ! -f "$CKPT_DIR/deco_best.pth" ]; then
        echo "  fetching DECO..."
        curl -fL "https://keeper.mpdl.mpg.de/f/6f2e2258558f46ceb269/?dl=1" \
            -o "$CKPT_DIR/deco_best.pth" \
            || warn "DECO download failed — grab manually if you intend to run --runner josh."
    fi

    # JOSH3R checkpoint — Google Drive
    if [ ! -f "$JOSH3R_CKPT_DIR/josh3r.pth" ]; then
        echo "  fetching JOSH3R..."
        "$VENV_PY" -m gdown \
            "https://drive.google.com/uc?id=1dlX4p1RfwFjuzjHr76SCl21QZAx1MLK4" \
            -O "$JOSH3R_CKPT_DIR/josh3r.pth" \
            || warn "JOSH3R download failed — grab manually from the link in JOSH/README."
    fi
else
    log "[6/7] SKIP_DOWNLOADS=1 set — skipping checkpoint download"
fi

# ----------------------------------------------------------- 7. Final SMPL check
log "[7/7] Verify SMPL assets"
SMPL_DIR="$VENDOR_DIR/JOSH/data/smpl"
mkdir -p "$SMPL_DIR"

needed=(SMPL_MALE.pkl SMPL_FEMALE.pkl SMPL_NEUTRAL.pkl)
missing=()
for f in "${needed[@]}"; do
    [ -f "$SMPL_DIR/$f" ] || missing+=("$f")
done
if [ "${#missing[@]}" -gt 0 ]; then
    warn "Missing SMPL pickles under $SMPL_DIR :"
    printf '       - %s\n' "${missing[@]}"
    echo   "       Register + download at https://smpl.is.tue.mpg.de then copy them in."
fi
if [ ! -f "$MODELS_DIR/smplx/SMPLX_NEUTRAL.npz" ]; then
    warn "Missing SMPLX_NEUTRAL.npz at $MODELS_DIR/smplx/ — needed by the viewer."
fi

# ------------------------------------------------------------------------ done
cat <<EOS

=== Setup complete ===

Next steps:
  export HSI_JOSH3R_CKPT="$REPO_ROOT/models/josh3r/josh3r.pth"
  export HSI_JOSH3R_SMPL_DIR="$VENDOR_DIR/JOSH/data/smpl"
  source "$VENV_ML/bin/activate"

Run the pipeline directly:
  cd "$SCRIPT_DIR"
  python scripts/run_pipeline.py /path/to/video.mp4 --runner josh3r

Or start the backend against this venv:
  HSI_RUNNER=josh3r HSI_DEVICE=cuda \\
    "$VENV_ML/bin/python" -m uvicorn app.main:app --app-dir "$REPO_ROOT/backend" \\
    --host 0.0.0.0 --port 8000

EOS
