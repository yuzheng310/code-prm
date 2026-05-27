#!/usr/bin/env bash
# Bootstrap a Linux + NVIDIA GPU box for Code-PRM development.
# Works on:
#   - Lab box (e.g. RTX 3090)
#   - AutoDL / cloud rental (e.g. L40S / RTX 6000 Ada / vGPU-48GB)
#
# Two modes (auto-detected):
#   A. SYSTEM mode  — base env already has PyTorch (typical rental image).
#                     Install our deps via pip into the base env. Fast.
#   B. CONDA  mode  — no PyTorch present; create a `code-prm` conda env from
#                     environment.yml. Slower (downloads PyTorch).
#
# Prerequisites BEFORE running:
#   1. SSH key pair: `ssh-keygen -t ed25519`
#   2. Public key in GitHub: `cat ~/.ssh/id_ed25519.pub`
#   3. Test:  `ssh -T git@github.com`  (should greet you)
#
# After SSH key is set up:
#   git clone git@github.com:yuzheng310/code-prm.git
#   cd code-prm
#   bash scripts/00_setup_lab_box.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> [1/5] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is this an NVIDIA GPU box?"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [ -z "$GPU_NAME" ]; then
    echo "ERROR: no NVIDIA GPU detected. nvidia-smi output:"
    nvidia-smi
    exit 1
fi
echo "    ✓ GPU detected: $GPU_NAME"

echo "==> [2/5] Checking disk space..."
FREE_GB=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$FREE_GB" -lt 50 ]; then
    echo "ERROR: need at least 50 GB free in \$HOME, only $FREE_GB GB available"
    exit 1
fi
echo "    ✓ $FREE_GB GB free in \$HOME"

echo "==> [3/5] Initializing submodules..."
git submodule update --init --recursive
echo "    ✓ submodules ready"

echo "==> [4/5] Detecting install mode..."
INSTALL_MODE=""
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "    ✓ system PyTorch found: $TORCH_VERSION  → using SYSTEM mode"
    INSTALL_MODE="system"
else
    if ! command -v conda &>/dev/null; then
        echo "    No conda found, installing Miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
        bash /tmp/mc.sh -b -p "$HOME/miniconda3"
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    echo "    ✓ no system PyTorch; will use CONDA mode"
    INSTALL_MODE="conda"
fi

echo "==> [5/5] Installing project deps ($INSTALL_MODE mode)..."
case "$INSTALL_MODE" in
    system)
        pip install -e . --quiet
        ;;
    conda)
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
        if conda env list | grep -q "^code-prm "; then
            echo "    env 'code-prm' already exists; updating"
        else
            conda env create -f environment.yml -n code-prm
        fi
        conda activate code-prm
        ;;
esac

echo ""
echo "==> Verifying installation..."
python -c "
import torch, transformers, peft, anthropic, pydantic
print(f'    PyTorch      {torch.__version__}')
print(f'    Transformers {transformers.__version__}')
print(f'    PEFT         {peft.__version__}')
print(f'    Anthropic    {anthropic.__version__}')
print(f'    Pydantic     {pydantic.__version__}')
assert torch.cuda.is_available(), 'CUDA not available'
print(f'    CUDA         {torch.version.cuda}')
print(f'    Device       {torch.cuda.get_device_name(0)}')
"

echo ""
echo "================================================================"
echo "Box ready (mode: $INSTALL_MODE)."
echo ""
echo "Next steps:"
echo "  1. Install tmux:    apt-get install -y tmux  (sudo if needed)"
echo "  2. Open session:    tmux new -s code-prm"
if [ "$INSTALL_MODE" = "conda" ]; then
    echo "  3. Activate env:    conda activate code-prm"
fi
echo "  4. Run tests:       pytest tests/ -v"
echo "  5. Proceed to plan Task 5+ (PRM800K download, etc.)"
echo "================================================================"
