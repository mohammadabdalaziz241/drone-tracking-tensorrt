"""Render a demo of the final pipeline on one VisDrone-MOT sequence: boxes, IDs, trails, FPS.

Writes an MP4 (full resolution, for posts) and a small GIF (for the README).

Example:
  python scripts/make_demo.py --seq uav0000117_02622_v --out docs/media/demo
"""

from __future__ import annotations

import argparse
import colorsys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

EVAL_CLASSES = [0, 3, 4, 5, 8]  # pedestrian, car, van, truck, bus
NAMES = {0: "pedestrian", 3: "car", 4: "van", 5: "truck", 8: "bus"}


def colour(track_id: int) -> tuple[int, int, int]:
    h = (track_id * 0.61803398875) % 1.0  # golden-ratio hues: neighbouring IDs look different
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", default="runs/detect/yolo11s_1280/weights/best_trt_fp16.engine")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--tracker", default="configs/trackers/botsort_t0.65.yaml")
    p.add_argument("--seq", default="uav0000117_02622_v")
    p.add_argument("--split", default="val")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--trail", type=int, default=20, help="trail length in frames")
    p.add_argument("--out", default="docs/media/demo", help="output path without extension")
    p.add_argument("--gif-width", type=int, default=640)
    p.add_argument("--gif-every", type=int, default=2, help="keep every Nth frame in the GIF")
    p.add_argument("--gif-frames", type=int, default=120)
    args = p.parse_args()

    seq_dir = Path(f"/scratch/ma06314/drone-tracking/datasets/VisDrone-MOT/VisDrone2019-MOT-{args.split}/sequences/{args.seq}")
    frames = sorted(seq_dir.glob("*.jpg"))[: args.max_frames]
    if not frames:
        raise SystemExit(f"no frames in {seq_dir}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.engine, task="detect")
    h, w = cv2.imread(str(frames[0])).shape[:2]
    video = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.trail))
    gif_frames: list[Image.Image] = []
    fps_hist: deque = deque(maxlen=30)

    for i, path in enumerate(frames):
        img = cv2.imread(str(path))
        t0 = time.perf_counter()
        r = model.track(img, persist=True, tracker=args.tracker, imgsz=args.imgsz,
                        classes=EVAL_CLASSES, verbose=False)[0]
        fps_hist.append(1.0 / (time.perf_counter() - t0))

        if r.boxes is not None and r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int),
                                                  r.boxes.cls.cpu().numpy().astype(int)):
                c = colour(tid)
                trails[tid].append((int((x1 + x2) / 2), int(y2)))
                pts = np.array(trails[tid], dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(img, [pts], False, c, 2, cv2.LINE_AA)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2, cv2.LINE_AA)
                label = f"{tid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (int(x1), int(y1) - th - 4), (int(x1) + tw + 4, int(y1)), c, -1)
                cv2.putText(img, label, (int(x1) + 2, int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        banner = (f"YOLO11s 1280 TensorRT FP16 + BoT-SORT (camera-motion comp.)  |  "
                  f"{np.median(fps_hist):.0f} FPS (Python, end to end)  |  frame {i + 1}")
        cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(img, banner, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        video.write(img)

        if i % args.gif_every == 0 and len(gif_frames) < args.gif_frames:
            small = cv2.resize(img, (args.gif_width, int(h * args.gif_width / w)), interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))

    video.release()
    gif_path = out.with_suffix(".gif")
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in gif_frames]
    pal[0].save(gif_path, save_all=True, append_images=pal[1:], duration=int(1000 * args.gif_every / 25),
                loop=0, optimize=True)
    print(f"MP4: {out.with_suffix('.mp4')}  ({len(frames)} frames)")
    print(f"GIF: {gif_path}  ({len(gif_frames)} frames, {gif_path.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
