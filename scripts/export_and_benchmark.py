"""Stage 2: export a trained detector and measure accuracy vs speed for each format.

Formats compared (same weights, same validation set, same input size):
  pytorch   - the trained .pt model (FP32)
  onnx      - ONNX Runtime on GPU (FP32)
  trt_fp16  - TensorRT engine, FP16
  trt_int8  - TensorRT engine, INT8 with calibration on VisDrone training images

For each format we record mAP50 / mAP50-95 (Ultralytics val) and batch-1 latency
measured with our own timing loop (warm-up, then median and p95 over N runs).

Example:
  python scripts/export_and_benchmark.py --weights runs/detect/yolo11s_640/weights/best.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def export(weights: Path, fmt: str, imgsz: int, data: str) -> Path:
    model = YOLO(str(weights))
    if fmt == "onnx":
        out = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
    elif fmt == "trt_fp16":
        out = model.export(format="engine", imgsz=imgsz, half=True, workspace=8)
    elif fmt == "trt_int8":
        out = model.export(format="engine", imgsz=imgsz, int8=True, data=data, fraction=0.25, workspace=8)
    else:
        raise ValueError(fmt)
    out = Path(out)
    # keep FP16 and INT8 engines from overwriting each other
    if fmt.startswith("trt"):
        renamed = out.with_name(f"{out.stem}_{fmt}{out.suffix}")
        out.rename(renamed)
        out = renamed
    return out


def latency_ms(model_path: Path, imgsz: int, images: list[str], warmup: int, runs: int) -> dict:
    model = YOLO(str(model_path), task="detect")
    for i in range(warmup):
        model.predict(images[i % len(images)], imgsz=imgsz, verbose=False)
    times = []
    for i in range(runs):
        img = images[i % len(images)]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = statistics.median(times)
    return {
        "end_to_end_median_ms": round(median, 2),
        "end_to_end_p95_ms": round(float(np.percentile(times, 95)), 2),
        "fps_batch1": round(1000 / median, 1),
    }


def evaluate(model_path: Path, imgsz: int, data: str) -> dict:
    model = YOLO(str(model_path), task="detect")
    m = model.val(data=data, imgsz=imgsz, batch=1, verbose=False, plots=False)
    return {
        "map50": round(float(m.box.map50), 4),
        "map50_95": round(float(m.box.map), 4),
        "ultralytics_speed_ms": {k: round(v, 2) for k, v in m.speed.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--data", default="VisDrone.yaml")
    p.add_argument("--formats", default="pytorch,onnx,trt_fp16,trt_int8")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--val-images", default="/scratch/ma06314/drone-tracking/datasets/VisDrone/images/val")
    args = p.parse_args()

    images = sorted(glob.glob(str(Path(args.val_images) / "*.jpg")))
    assert images, f"no images found in {args.val_images}"

    run = args.weights.parents[1].name  # e.g. yolo11s_640
    rows = []
    for fmt in args.formats.split(","):
        print(f"\n=== {fmt} ===", flush=True)
        path = args.weights if fmt == "pytorch" else export(args.weights, fmt, args.imgsz, args.data)
        row = {"format": fmt, "file": str(path), "size_mb": round(path.stat().st_size / 2**20, 2)}
        row.update(evaluate(path, args.imgsz, args.data))
        row.update(latency_ms(path, args.imgsz, images, args.warmup, args.runs))
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    base = rows[0]
    for r in rows:
        r["map50_95_change_vs_pytorch"] = round(r["map50_95"] - base["map50_95"], 4)
        r["speedup_vs_pytorch"] = round(base["end_to_end_median_ms"] / r["end_to_end_median_ms"], 2)

    out = {
        "run": run,
        "imgsz": args.imgsz,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timing": f"batch 1, {args.warmup} warm-up + {args.runs} timed predictions on VisDrone val images, end to end (pre-process + inference + NMS)",
        "results": rows,
    }
    dest = ROOT / "results" / f"stage2_{run}.json"
    dest.write_text(json.dumps(out, indent=2))

    print("\n| Format | Size (MB) | mAP50-95 | Δ vs PyTorch | Median ms | p95 ms | FPS | Speed-up |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['format']} | {r['size_mb']} | {r['map50_95']:.4f} | {r['map50_95_change_vs_pytorch']:+.4f} | "
              f"{r['end_to_end_median_ms']} | {r['end_to_end_p95_ms']} | {r['fps_batch1']} | {r['speedup_vs_pytorch']}x |")
    print(f"\nSaved: {dest}")


if __name__ == "__main__":
    main()
