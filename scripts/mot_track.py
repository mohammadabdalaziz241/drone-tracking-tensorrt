"""Run detector + tracker over every VisDrone-MOT sequence and write MOTChallenge results.

Only the five evaluated categories are tracked (pedestrian, car, van, truck, bus).
Predictions mostly inside an annotated "ignored region" are removed, as in the official toolkit.

Example:
  python scripts/mot_track.py --engine runs/detect/yolo11s_1280/weights/best_trt_fp16.engine \
      --imgsz 1280 --tracker botsort.yaml --name y1280_botsort
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# Detector class indices (VisDrone-DET order) of the five evaluated MOT categories.
EVAL_CLASSES = [0, 3, 4, 5, 8]  # pedestrian, car, van, truck, bus


def inside_fraction(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """For each xyxy box, the largest fraction of its area covered by one xywh region."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes))
    r = np.concatenate([regions[:, :2], regions[:, :2] + regions[:, 2:]], axis=1)
    iw = np.clip(np.minimum(boxes[:, None, 2], r[None, :, 2]) - np.maximum(boxes[:, None, 0], r[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(boxes[:, None, 3], r[None, :, 3]) - np.maximum(boxes[:, None, 1], r[None, :, 1]), 0, None)
    area = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6, None)
    return (iw * ih / area[:, None]).max(axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True)
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--tracker", required=True, help="bytetrack.yaml or botsort.yaml")
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="val", help="val or test-dev")
    p.add_argument("--visdrone", default=None, help="defaults to datasets/VisDrone-MOT/VisDrone2019-MOT-<split>")
    p.add_argument("--eval-root", default="/scratch/ma06314/drone-tracking/mot_eval")
    p.add_argument("--ignore-frac", type=float, default=0.5)
    args = p.parse_args()

    visdrone = args.visdrone or f"/scratch/ma06314/drone-tracking/datasets/VisDrone-MOT/VisDrone2019-MOT-{args.split}"
    seq_root = Path(visdrone) / "sequences"
    eval_root = Path(args.eval_root)
    ignored = json.loads((eval_root / f"ignored_regions_{args.split}.json").read_text())
    split_dir = eval_root / "trackers" / f"VisDrone-{args.split}"
    out_dir = split_dir / args.name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames, total_time = 0, 0.0
    for seq in sorted(d.name for d in seq_root.iterdir() if d.is_dir()):
        model = YOLO(args.engine, task="detect")  # fresh tracker state per sequence
        frames = sorted((seq_root / seq).glob("*.jpg"))
        lines = []
        t0 = time.perf_counter()
        for fi, path in enumerate(frames, start=1):
            r = model.track(str(path), persist=True, tracker=args.tracker, imgsz=args.imgsz,
                            classes=EVAL_CLASSES, verbose=False)[0]
            if r.boxes is None or r.boxes.id is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            regions = np.array(ignored.get(seq, {}).get(str(fi), []), dtype=float).reshape(-1, 4)
            keep = inside_fraction(xyxy, regions) <= args.ignore_frac
            for (x1, y1, x2, y2), tid, c in zip(xyxy[keep], ids[keep], conf[keep]):
                lines.append(f"{fi},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f},-1,-1,-1\n")
        dt = time.perf_counter() - t0
        (out_dir / f"{seq}.txt").write_text("".join(lines))
        total_frames += len(frames)
        total_time += dt
        print(f"{args.name} {seq}: {len(frames)} frames, {len(frames) / dt:.1f} FPS", flush=True)

    summary = {"name": args.name, "split": args.split, "engine": args.engine, "imgsz": args.imgsz, "tracker": args.tracker,
               "frames": total_frames, "fps_end_to_end_python": round(total_frames / total_time, 1)}
    (split_dir / args.name / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
