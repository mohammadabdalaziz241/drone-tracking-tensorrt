#!/usr/bin/env bash
# Stage 1: train the 640 baseline, then the 1280 small-object model.
# Re-runnable: a finished run (results/<name>.json exists) is skipped,
# an interrupted run resumes from its last.pt, otherwise it starts fresh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/scratch/ma06314/drone-tracking/.venv/bin/python}"
mkdir -p "$ROOT/logs"
cd "$ROOT"

run() {  # name imgsz batch
  local name=$1 imgsz=$2 batch=$3
  if [[ -f "results/${name}.json" ]]; then
    echo "[$name] already finished, skipping"; return
  fi
  local extra=()
  [[ -f "runs/detect/${name}/weights/last.pt" ]] && extra=(--resume)
  echo "[$name] starting ${extra[*]:-fresh} at $(date)"
  "$PY" scripts/train.py --imgsz "$imgsz" --batch "$batch" --name "$name" "${extra[@]}" 2>&1 | tee -a "logs/${name}.log"
}

run yolo11s_640  640  16
run yolo11s_1280 1280 8
echo "Stage 1 finished: $(date)"
