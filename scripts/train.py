"""Fine-tune a YOLO detector on VisDrone-DET.

Two configurations form the first experiment:
  * baseline:     imgsz 640  (the default most tutorials use)
  * small-object: imgsz 1280 (more pixels per tiny object)

Example:
  python scripts/train.py --imgsz 640  --name yolo11s_640
  python scripts/train.py --imgsz 1280 --name yolo11s_1280 --batch 8
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo11s.pt", help="pretrained COCO checkpoint to start from")
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", required=True, help="run name, e.g. yolo11s_640")
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--resume", action="store_true", help="continue from <project>/<name>/weights/last.pt")
    args = parser.parse_args()

    started = time.time()
    last = Path(args.project) / args.name / "weights" / "last.pt"
    if args.resume and last.exists():
        print(f"Resuming from {last}", flush=True)
        results = YOLO(str(last)).train(resume=True)
    else:
        results = YOLO(args.model).train(
            data=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            patience=30,
            cos_lr=True,
            close_mosaic=10,
            project=args.project,
            name=args.name,
            exist_ok=True,
            plots=True,
        )

    run_dir = Path(results.save_dir)
    summary = {
        "run": args.name,
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "per_class_map50_95": {
            results.names[i]: float(v) for i, v in enumerate(results.box.maps)
        },
        "train_hours_this_session": round((time.time() - started) / 3600, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    out = ROOT / "results" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"Summary: {out}")


if __name__ == "__main__":
    main()
