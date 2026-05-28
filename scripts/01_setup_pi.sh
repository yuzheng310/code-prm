#!/usr/bin/env bash
# Set up `pi` (the coding agent harness) on the lab box and install our
# trajectory_logger extension.
#
# Pi: https://github.com/earendil-works/pi (or your fork)
# We use the fork yuzheng310/pi to pin a specific commit.
#
# Run from project root after scripts/00_setup_lab_box.sh.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PI_REPO_URL="${PI_REPO_URL:-git@github.com:yuzheng310/pi.git}"
PI_INSTALL_DIR="${PI_INSTALL_DIR:-$HOME/pi}"
EXTENSION_SRC="$PROJECT_DIR/src/collector/trajectory_logger.ts"

echo "==> [1/4] Cloning pi to $PI_INSTALL_DIR..."
if [ ! -d "$PI_INSTALL_DIR" ]; then
    git clone "$PI_REPO_URL" "$PI_INSTALL_DIR"
else
    echo "    pi already cloned; pulling latest"
    (cd "$PI_INSTALL_DIR" && git pull --ff-only) || true
fi

echo "==> [2/4] Installing pi npm deps + building..."
cd "$PI_INSTALL_DIR"
# Use the project's pinned npm if available.
if [ -f package-lock.json ]; then
    npm ci --no-audit --no-fund
else
    npm install --no-audit --no-fund
fi
npm run build

CLI_PATH="$PI_INSTALL_DIR/packages/coding-agent/dist/cli.js"
if [ ! -f "$CLI_PATH" ]; then
    echo "ERROR: expected built CLI at $CLI_PATH but it doesn't exist."
    exit 1
fi
echo "    ✓ pi CLI built at $CLI_PATH"

echo "==> [3/4] Symlinking trajectory_logger.ts into pi extensions dir..."
GLOBAL_EXT_DIR="$HOME/.pi/agent/extensions"
mkdir -p "$GLOBAL_EXT_DIR"
ln -sf "$EXTENSION_SRC" "$GLOBAL_EXT_DIR/trajectory_logger.ts"
echo "    ✓ symlink created: $GLOBAL_EXT_DIR/trajectory_logger.ts → $EXTENSION_SRC"

echo "==> [4/4] Updating .env hint for TS_REPO_PATH..."
echo ""
echo "Add (or update) in your shell env or .env file:"
echo ""
echo "    export TS_REPO_PATH=\"$PI_INSTALL_DIR/packages/coding-agent\""
echo ""
echo "(This is what src/eval/swebench_runner.py uses to locate dist/cli.js.)"
echo ""
echo "================================================================"
echo "Pi ready. Smoke-test the extension is loaded:"
echo "  node $CLI_PATH --help"
echo ""
echo "Then the pilot collection:"
echo "  export ANTHROPIC_API_KEY=..."
echo "  export TS_REPO_PATH=\"$PI_INSTALL_DIR/packages/coding-agent\""
echo "  bash scripts/05_collect_pilot.sh"
echo "================================================================"
