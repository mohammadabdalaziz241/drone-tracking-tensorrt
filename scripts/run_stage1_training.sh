#!/usr/bin/env bash
# Stage 1: train the 640 baseline, then the 1280 small-object model, one after the other.
# Safe to start with nohup; logs go to logs/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/scratch/ma06314/drone-tracking/.venv/bin/python}"
mkdir -p "$ROOT/logs"
cd "$ROOT"

"$PY" scripts/train.py --imgsz 640  --batch 16 --name yolo11s_640  2>&1 | tee "logs/yolo11s_640.log"
"$PY" scripts/train.py --imgsz 1280 --batch 8  --name yolo11s_1280 2>&1 | tee "logs/yolo11s_1280.log"
echo "Stage 1 finished: $(date)"
