"""Check that the C++ engine produces the same detections as the Python (Ultralytics) path.

Both run the SAME TensorRT engine on the same images with the same thresholds, so any
difference comes from pre-processing (our CUDA letterbox kernel vs OpenCV) or post-processing
(our NMS vs Ultralytics'). Agreement is reported at several IoU thresholds: a strict one
(0.9) is sensitive to one-pixel box jitter on tiny objects, a loose one (0.5) only counts
genuinely missing or extra detections.

Example:
  build/trt_bench --engine E --images DIR --max-images 100 --runs 1 --warmup 0 --dump cpp.json
  python scripts/parity_check.py --engine E --images DIR --cpp cpp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MATCH_IOUS = (0.5, 0.75, 0.9)


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


def match(ref: np.ndarray, other: np.ndarray, thr: float) -> int:
    """Greedy one-to-one matching, same class, highest-confidence reference boxes first."""
    if len(ref) == 0 or len(other) == 0:
        return 0
    m = iou_matrix(ref[:, :4], other[:, :4])
    m[ref[:, 5][:, None] != other[:, 5][None, :]] = 0
    taken = np.zeros(len(other), dtype=bool)
    n = 0
    for i in np.argsort(-ref[:, 4]):
        row = np.where(taken, -1.0, m[i])
        j = int(np.argmax(row))
        if row[j] >= thr:
            taken[j] = True
            n += 1
    return n


def to_array(boxes) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 6))
    return np.concatenate([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None],
                           boxes.cls.cpu().numpy()[:, None]], axis=1)


def agreement(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    n_ref = sum(len(a) for a, _ in pairs)
    n_oth = sum(len(b) for _, b in pairs)
    out = {"reference_detections": n_ref, "other_detections": n_oth}
    for thr in MATCH_IOUS:
        k = sum(match(a, b, thr) for a, b in pairs)
        out[f"recall@iou{thr}"] = round(k / max(n_ref, 1), 4)
        out[f"precision@iou{thr}"] = round(k / max(n_oth, 1), 4)
    return out


def main() -> None:
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--cpp", required=True, help="JSON written by trt_bench --dump")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cpp = json.loads(Path(args.cpp).read_text())
    model = YOLO(args.engine, task="detect")
    pairs = []
    for name, dets in cpp.items():
        r = model.predict(str(Path(args.images) / name), conf=args.conf, iou=args.iou,
                          max_det=300, agnostic_nms=False, verbose=False)[0]
        pairs.append((to_array(r.boxes), np.array(dets, dtype=float).reshape(-1, 6)))

    summary = {"engine": args.engine, "images": len(cpp), "reference": "python", **agreement(pairs)}
    # Backwards-compatible keys used by run_stage3.sh's table.
    summary["recall_of_python_by_cpp"] = summary["recall@iou0.9"]
    summary["precision_of_cpp_vs_python"] = summary["precision@iou0.9"]
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
