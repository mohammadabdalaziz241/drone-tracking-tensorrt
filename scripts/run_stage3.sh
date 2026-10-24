#!/usr/bin/env bash
# Stage 3: C++ benchmarks + C++/Python parity for every TensorRT engine from Stage 2.
# Run only when the GPU is otherwise idle, or the timings are meaningless.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/scratch/ma06314/drone-tracking/.venv/bin/python}"
VAL="${VAL:-/scratch/ma06314/drone-tracking/datasets/VisDrone/images/val}"
cd "$ROOT"
mkdir -p results/stage3 logs

for run in yolo11s_640 yolo11s_1280; do
  for prec in fp16 int8; do
    engine="runs/detect/${run}/weights/best_trt_${prec}.engine"
    [[ -f "$engine" ]] || { echo "skip: $engine not found"; continue; }
    tag="${run}_${prec}"
    echo "=== ${tag} ==="
    build/trt_bench --engine "$engine" --images "$VAL" --max-images 100 --warmup 50 --runs 300 \
      --out "results/stage3/bench_${tag}.json"
    build/trt_bench --engine "$engine" --images "$VAL" --max-images 100 --warmup 0 --runs 1 \
      --dump "results/stage3/cpp_dets_${tag}.json" --vis "results/stage3/vis_${tag}" > /dev/null
    "$PY" scripts/parity_check.py --engine "$engine" --images "$VAL" \
      --cpp "results/stage3/cpp_dets_${tag}.json" --out "results/stage3/parity_${tag}.json"
  done
done

"$PY" - <<'EOF'
import json, glob, os
rows = []
for f in sorted(glob.glob("results/stage3/bench_*.json")):
    tag = os.path.basename(f)[6:-5]
    b = json.load(open(f)); m = b["median_ms"]
    par = json.load(open(f"results/stage3/parity_{tag}.json"))
    rows.append((tag, m, b["p95_total_ms"], b["fps_batch1"], par["recall_of_python_by_cpp"], par["precision_of_cpp_vs_python"]))
print("\n| Engine | Upload | Pre-proc (CUDA) | Inference | Download | Post-proc | Total | p95 | FPS | Parity R / P |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for t, m, p95, fps, r, p in rows:
    print(f"| {t} | {m['upload']:.2f} | {m['preprocess']:.2f} | {m['inference']:.2f} | {m['download']:.2f} | "
          f"{m['postprocess']:.2f} | {m['total']:.2f} | {p95:.2f} | {fps:.0f} | {r:.3f} / {p:.3f} |")
EOF
