#!/usr/bin/env bash
# Stage 4c: one-shot confirmation on VisDrone2019-MOT test-dev.
#
# Everything below was fixed BEFORE looking at test-dev:
#  - the four default-threshold ablation runs (detector 640/1280 x ByteTrack/BoT-SORT);
#  - the final configuration, selected on val by maximum HOTA: 1280 + BoT-SORT, thresholds 0.65;
#  - a secondary control, 640 + BoT-SORT at 0.65, to separate the detector's contribution from
#    the threshold tuning.
# Run it once; do not tune anything on these numbers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/scratch/ma06314/drone-tracking/.venv/bin/python}"
SPLIT=test-dev
cd "$ROOT"
mkdir -p results/stage4

"$PY" scripts/mot_prepare_gt.py --split "$SPLIT"
"$PY" scripts/make_tracker_cfg.py --base botsort.yaml --out configs/trackers/botsort_t0.65.yaml \
  track_high_thresh=0.65 new_track_thresh=0.65

run() {  # name engine-size tracker-cfg
  "$PY" scripts/mot_track.py --split "$SPLIT" --engine "runs/detect/yolo11s_$2/weights/best_trt_fp16.engine" \
    --imgsz "$2" --tracker "$3" --name "$1"
}
run y640_bytetrack       640  bytetrack.yaml
run y640_botsort         640  botsort.yaml
run y1280_bytetrack      1280 bytetrack.yaml
run y1280_botsort        1280 botsort.yaml
run y640_botsort_t0.65   640  configs/trackers/botsort_t0.65.yaml
run y1280_botsort_t0.65  1280 configs/trackers/botsort_t0.65.yaml

"$PY" scripts/mot_eval.py --split "$SPLIT" --out results/stage4/testdev.json --trackers \
  y640_bytetrack y640_botsort y1280_bytetrack y1280_botsort y640_botsort_t0.65 y1280_botsort_t0.65 2>/dev/null
