#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Create venv and install deps if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ ! -f "$VENV_DIR/bin/blenderproc" ]; then
    echo "Installing blenderproc..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Install Pillow for PNG export (into the system Python, used by blenderproc's bundled Blender)
pip install Pillow 2>/dev/null || true

# Forward all arguments to the generation script
echo "Starting dataset generation..."
blenderproc run "$SCRIPT_DIR/generate_dataset.py" "$@"
