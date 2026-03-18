#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Parallel dataset generation launcher
#
# Splits the total scene count across N BlenderProc worker processes, each
# handling a non-overlapping range of scene indices.  All workers write to
# the same output directory (scene indices never collide).
#
# Usage:
#   ./generate_parallel.sh [OPTIONS]
#
# Options (all optional):
#   --num-scenes   N   Total scenes to generate      (default: 1000)
#   --views-per-scene N  Camera views per scene       (default: 4)
#   --workers      N   Number of parallel workers     (default: 4)
#   --output-dir   DIR Output directory               (default: /workspace/beer_pong_dataset)
#   --seed         N   Base random seed               (default: 42)
#   --assets-dir   DIR Path to downloaded assets      (default: ./assets)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# --- Defaults ---
NUM_SCENES=1000
VIEWS_PER_SCENE=4
WORKERS=4
OUTPUT_DIR="/workspace/beer_pong_dataset"
SEED=42
ASSETS_DIR="$SCRIPT_DIR/assets"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-scenes)      NUM_SCENES="$2";      shift 2 ;;
        --views-per-scene) VIEWS_PER_SCENE="$2"; shift 2 ;;
        --workers)         WORKERS="$2";         shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --seed)            SEED="$2";            shift 2 ;;
        --assets-dir)      ASSETS_DIR="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Environment setup (once) ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ ! -f "$VENV_DIR/bin/blenderproc" ]; then
    echo "Installing blenderproc..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

pip install Pillow 2>/dev/null || true

# --- Compute per-worker scene ranges ---
SCENES_PER_WORKER=$((NUM_SCENES / WORKERS))
REMAINDER=$((NUM_SCENES % WORKERS))

echo "============================================================"
echo " Parallel Dataset Generation"
echo "============================================================"
echo " Total scenes:    $NUM_SCENES"
echo " Views per scene: $VIEWS_PER_SCENE"
echo " Total images:    $((NUM_SCENES * VIEWS_PER_SCENE))"
echo " Workers:         $WORKERS"
echo " Scenes/worker:   $SCENES_PER_WORKER (+$REMAINDER remainder)"
echo " Output:          $OUTPUT_DIR"
echo " Seed:            $SEED"
echo " Assets:          $ASSETS_DIR"
echo "============================================================"

# Create output dirs ahead of time (avoid race in workers)
mkdir -p "$OUTPUT_DIR/images" "$OUTPUT_DIR/labels"

# Create a directory for per-worker logs
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# --- Launch workers ---
PIDS=()
START=0

for ((w=0; w<WORKERS; w++)); do
    # Distribute remainder scenes across the first few workers
    COUNT=$SCENES_PER_WORKER
    if [ $w -lt $REMAINDER ]; then
        COUNT=$((COUNT + 1))
    fi

    if [ $COUNT -eq 0 ]; then
        continue
    fi

    LOG_FILE="$LOG_DIR/worker_${w}.log"

    echo "  Worker $w: scenes $START..$((START + COUNT - 1)) ($COUNT scenes) -> $LOG_FILE"

    blenderproc run "$SCRIPT_DIR/generate_dataset.py" \
        --num-scenes "$COUNT" \
        --views-per-scene "$VIEWS_PER_SCENE" \
        --output-dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        --start-scene "$START" \
        --assets-dir "$ASSETS_DIR" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    START=$((START + COUNT))
done

echo ""
echo "Launched ${#PIDS[@]} workers. Waiting for completion..."
echo "  Monitor progress:  tail -f $LOG_DIR/worker_*.log"
echo "  Monitor GPU:       watch -n1 nvidia-smi"
echo ""

# --- Wait for all workers, track failures ---
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    if wait "$pid"; then
        echo "  Worker $i (PID $pid) completed successfully."
    else
        EXIT_CODE=$?
        echo "  Worker $i (PID $pid) FAILED with exit code $EXIT_CODE."
        echo "  Check log: $LOG_DIR/worker_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    TOTAL_IMAGES=$(find "$OUTPUT_DIR/images" -name '*.png' 2>/dev/null | wc -l)
    echo "All workers completed successfully!"
    echo "Total images generated: $TOTAL_IMAGES"
    echo "Output: $OUTPUT_DIR"
else
    echo "$FAILED worker(s) failed. Check logs in $LOG_DIR/"
    exit 1
fi
