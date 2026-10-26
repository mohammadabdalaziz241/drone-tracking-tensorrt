#!/usr/bin/env bash
# Stage 4: multi-object tracking on VisDrone2019-MOT val.
# 2 detectors (640, 1280; TensorRT FP16) x 2 trackers (ByteTrack, BoT-SORT with camera-motion
# compensation), evaluated with TrackEval (HOTA / MOTA / IDF1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/scratch/ma06314/drone-tracking/.venv/bin/python}"
cd "$ROOT"
mkdir -p results/stage4

# TrackEval (not on PyPI); patch the NumPy aliases removed in NumPy >= 1.24.
if ! "$PY" -c "import trackeval" 2>/dev/null; then
  "$PY" -m pip install -q "git+https://github.com/JonathonLuiten/TrackEval.git"
fi
TE="$("$PY" -c 'import trackeval, os; print(os.path.dirname(trackeval.__file__))')"
grep -rlE "np\.(float|int|bool)\b" "$TE" | xargs -r sed -i -E 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g'

"$PY" scripts/mot_prepare_gt.py

for size in 640 1280; do
  for trk in bytetrack botsort; do
    name="y${size}_${trk}"
    "$PY" scripts/mot_track.py --engine "runs/detect/yolo11s_${size}/weights/best_trt_fp16.engine" \
      --imgsz "$size" --tracker "${trk}.yaml" --name "$name"
  done
done

"$PY" scripts/mot_eval.py --out results/stage4/tracking.json \
  --trackers y640_bytetrack y640_botsort y1280_bytetrack y1280_botsort
