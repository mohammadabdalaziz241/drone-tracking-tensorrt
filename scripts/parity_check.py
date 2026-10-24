"""Check that the C++ engine produces the same detections as the Python (Ultralytics) path.

Both run the SAME TensorRT engine on the same images with the same thresholds, so any
difference comes from pre-processing (our CUDA letterbox kernel vs OpenCV) or post-processing
(our NMS vs Ultralytics'). A detection counts as matched if a box of the same class overlaps
with IoU >= --match-iou.

Example:
  build/trt_bench --engine E --images DIR --max-images 100 --runs 1 --warmup 0 --dump cpp.json
  python scripts/parity_check.py --engine E --images DIR --cpp cpp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--cpp", required=True, help="JSON written by trt_bench --dump")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--match-iou", type=float, default=0.9)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cpp = json.loads(Path(args.cpp).read_text())
    model = YOLO(args.engine, task="detect")
    matched = py_total = cpp_total = 0
    score_diffs, ious = [], []
    for name, cpp_dets in cpp.items():
        r = model.predict(str(Path(args.images) / name), conf=args.conf, iou=args.iou,
                          max_det=300, agnostic_nms=False, verbose=False)[0]
        py = np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None],
                             r.boxes.cls.cpu().numpy()[:, None]], axis=1) if len(r.boxes) else np.zeros((0, 6))
        cc = np.array(cpp_dets, dtype=float).reshape(-1, 6)
        py_total += len(py)
        cpp_total += len(cc)
        m = iou_matrix(py[:, :4], cc[:, :4])
        m[py[:, 5][:, None] != cc[:, 5][None, :]] = 0  # same class only
        used = set()
        for i in np.argsort(-py[:, 4]):
            if m.shape[1] == 0:
                break
            j = int(np.argmax(np.where([k in used for k in range(m.shape[1])], -1, m[i])))
            if m[i, j] >= args.match_iou and j not in used:
                used.add(j)
                matched += 1
                ious.append(m[i, j])
                score_diffs.append(abs(py[i, 4] - cc[j, 4]))

    summary = {
        "engine": args.engine,
        "images": len(cpp),
        "python_detections": py_total,
        "cpp_detections": cpp_total,
        "matched": matched,
        "recall_of_python_by_cpp": round(matched / max(py_total, 1), 4),
        "precision_of_cpp_vs_python": round(matched / max(cpp_total, 1), 4),
        "mean_iou_of_matches": round(float(np.mean(ious)), 4) if ious else None,
        "mean_abs_score_diff": round(float(np.mean(score_diffs)), 5) if score_diffs else None,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
