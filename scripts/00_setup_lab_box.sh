#!/usr/bin/env bash
# Bootstrap the lab box (Linux + RTX 3090) for Code-PRM development.
# Run this ON the lab box, after `ssh <labbox>` and `git clone`.
#
# Prerequisites on lab box BEFORE running this:
#   1. SSH key pair generated: `ssh-keygen -t ed25519`
#   2. Public key added to GitHub: cat ~/.ssh/id_ed25519.pub | pbcopy
#      then add at https://github.com/settings/keys
#   3. Test: `ssh -T git@github.com`  (should greet you by name)
#
# After SSH key is set up, on lab box:
#   git clone git@github.com:yuzheng310/code-prm.git
#   cd code-prm
#   bash scripts/00_setup_lab_box.sh

set -euo pipefail

REPO_URL="git@github.com:yuzheng310/code-prm.git"
PROJECT_DIR="$HOME/code-prm"

echo "==> [1/6] Checking GPU..."
if ! nvidia-smi | grep -q "RTX 3090"; then
    echo "ERROR: no RTX 3090 detected. nvidia-smi output:"
    nvidia-smi
    exit 1
fi
echo "    ✓ RTX 3090 found"

echo "==> [2/6] Checking disk space..."
FREE_GB=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$FREE_GB" -lt 100 ]; then
    echo "ERROR: need at least 100 GB free in \$HOME, only $FREE_GB GB available"
    exit 1
fi
echo "    ✓ $FREE_GB GB free in \$HOME"

echo "==> [3/6] Installing Miniconda if missing..."
if ! command -v conda &>/dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$HOME/miniconda3"
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/bin/conda" init bash
    echo "    ✓ Miniconda installed; you may need to re-source ~/.bashrc"
else
    echo "    ✓ conda already installed: $(conda --version)"
fi

# Make conda available in this script even if not initialized in shell yet
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

echo "==> [4/6] Cloning repo (if not already cloned)..."
if [ ! -d "$PROJECT_DIR" ]; then
    cd "$HOME"
    git clone "$REPO_URL" code-prm
fi
cd "$PROJECT_DIR"
git submodule update --init --recursive
echo "    ✓ repo at $PROJECT_DIR with submodules"

echo "==> [5/6] Creating conda env..."
if conda env list | grep -q "^code-prm "; then
    echo "    ✓ env 'code-prm' already exists; skipping create"
else
    conda env create -f environment.yml -n code-prm
    echo "    ✓ env 'code-prm' created"
fi

echo "==> [6/6] Verifying CUDA + PyTorch..."
conda activate code-prm
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'    PyTorch {torch.__version__}')
print(f'    CUDA   {torch.version.cuda}')
print(f'    Device {torch.cuda.get_device_name(0)}')
"

echo ""
echo "================================================================"
echo "Lab box ready. Next steps:"
echo "  1. Install tmux:    sudo apt-get install -y tmux"
echo "  2. Open session:    tmux new -s code-prm"
echo "  3. Inside tmux, activate env:  conda activate code-prm"
echo "  4. Run Task 5+ from the plan."
echo "================================================================"
