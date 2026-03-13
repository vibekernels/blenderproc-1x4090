#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Create venv and install blenderproc if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ ! -f "$VENV_DIR/bin/blenderproc" ]; then
    echo "Installing blenderproc..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Run the render
echo "Starting render..."
blenderproc run "$SCRIPT_DIR/render_beer_pong.py"

# Extract PNGs from HDF5 output
echo "Extracting PNGs..."
python3 -c "
import h5py, os, glob
from PIL import Image
import numpy as np

output_dir = os.path.join('$SCRIPT_DIR', 'output')
names = ['end_view', 'side_view', 'corner_view', 'far_end_view']

for i, name in enumerate(names):
    path = os.path.join(output_dir, f'{i}.hdf5')
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            img = Image.fromarray(np.array(f['colors']))
            img.save(os.path.join(output_dir, f'{name}.png'))
            print(f'  Saved {name}.png')
"

echo "Done! Output in $SCRIPT_DIR/output/"
