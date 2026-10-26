#!/usr/bin/env bash
# Stage 4b: tune the tracker's confidence thresholds for the 1280 detector + BoT-SORT.
# track_high_thresh (first-stage association) and new_track_thresh (track birth) are raised
# together; 0.25 is the Ultralytics default already evaluated as y1280_botsort.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/scratch/ma06314/drone-tracking/.venv/bin/python}"
cd "$ROOT"

names=(y1280_botsort)
for t in 0.35 0.45 0.55 0.65; do
  cfg="configs/trackers/botsort_t${t}.yaml"
  "$PY" scripts/make_tracker_cfg.py --base botsort.yaml --out "$cfg" track_high_thresh=$t new_track_thresh=$t
  name="y1280_botsort_t${t}"
  "$PY" scripts/mot_track.py --engine runs/detect/yolo11s_1280/weights/best_trt_fp16.engine \
    --imgsz 1280 --tracker "$cfg" --name "$name" --split val
  names+=("$name")
done

"$PY" scripts/mot_eval.py --split val --out results/stage4/threshold_sweep.json --trackers "${names[@]}" 2>/dev/null
