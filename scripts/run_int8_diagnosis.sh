#!/usr/bin/env bash
# Diagnose the INT8 C++/Python parity gap:
#  1) re-score the existing C++ dumps at IoU 0.5 / 0.75 / 0.9
#  2) measure each engine's sensitivity to +-1 pixel-value noise (Python path only)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/scratch/ma06314/drone-tracking/.venv/bin/python}"
VAL="${VAL:-/scratch/ma06314/drone-tracking/datasets/VisDrone/images/val}"
cd "$ROOT"

echo "| Engine | C++ vs Python @IoU0.5 | @IoU0.75 | @IoU0.9 |"
echo "|---|---:|---:|---:|"
for run in yolo11s_640 yolo11s_1280; do
  for prec in fp16 int8; do
    tag="${run}_${prec}"
    engine="runs/detect/${run}/weights/best_trt_${prec}.engine"
    "$PY" scripts/parity_check.py --engine "$engine" --images "$VAL" \
      --cpp "results/stage3/cpp_dets_${tag}.json" --out "results/stage3/parity_${tag}.json" > /dev/null
    "$PY" -c "import json;r=json.load(open('results/stage3/parity_${tag}.json'));print(f\"| ${tag} | {r['recall@iou0.5']:.3f} | {r['recall@iou0.75']:.3f} | {r['recall@iou0.9']:.3f} |\")"
  done
done

echo
echo "Noise sensitivity (same engine, original vs +-1 noise):"
"$PY" scripts/int8_sensitivity.py --images "$VAL" --out results/stage3/noise_sensitivity.json --engines \
  runs/detect/yolo11s_640/weights/best_trt_fp16.engine runs/detect/yolo11s_640/weights/best_trt_int8.engine \
  runs/detect/yolo11s_1280/weights/best_trt_fp16.engine runs/detect/yolo11s_1280/weights/best_trt_int8.engine \
  | sed -n '/^| Engine/,$p'
