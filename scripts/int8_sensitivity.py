"""How sensitive is each engine to imperceptible input noise?

For every image we run the engine on the original and on a copy with random integer noise in
{-1, 0, +1} added to each pixel value (invisible to the eye), then measure how many detections
survive, at several IoU thresholds. A robust model should be nearly unchanged; a fragile one
(e.g. a poorly calibrated INT8 engine) changes many detections.

Example:
  python scripts/int8_sensitivity.py --engines E_fp16 E_int8 --images DIR
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from parity_check import agreement, to_array


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engines", nargs="+", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    paths = sorted(glob.glob(str(Path(args.images) / "*.jpg")))[: args.max_images]
    rng = np.random.default_rng(args.seed)
    images = [cv2.imread(f) for f in paths]
    noisy = [np.clip(im.astype(np.int16) + rng.integers(-1, 2, im.shape), 0, 255).astype(np.uint8)
             for im in images]

    results = {}
    for engine in args.engines:
        model = YOLO(engine, task="detect")
        kw = dict(conf=args.conf, iou=args.iou, max_det=300, agnostic_nms=False, verbose=False)
        pairs = [(to_array(model.predict(a, **kw)[0].boxes), to_array(model.predict(b, **kw)[0].boxes))
                 for a, b in zip(images, noisy)]
        results[engine] = agreement(pairs)
        print(engine, json.dumps(results[engine], indent=2), flush=True)

    print("\n| Engine | Agreement @IoU0.5 | @IoU0.75 | @IoU0.9 |")
    print("|---|---:|---:|---:|")
    for e, r in results.items():
        print(f"| {Path(e).parents[1].name}/{Path(e).stem} | {r['recall@iou0.5']:.3f} | "
              f"{r['recall@iou0.75']:.3f} | {r['recall@iou0.9']:.3f} |")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
