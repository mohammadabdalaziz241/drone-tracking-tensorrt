# Drone small-object detection and tracking with TensorRT

> **Status:** stages 1–4 complete (detection, TensorRT export, C++/CUDA engine, tracking confirmed on a held-out test set). Stage 5, a multi-threaded C++ tracking pipeline, is in progress.

Detecting and tracking small objects (pedestrians, vehicles) in aerial drone video is hard: objects are often under 32×32 pixels, crowded, and seen from unusual angles. This project builds the full pipeline, from fine-tuning a detector to a deployed C++/CUDA TensorRT engine, and measures what every step gains and costs.

## Headline

**Tracking (VisDrone2019-MOT test-dev, held out until the end):** the final pipeline (YOLO11s at 1280 px as a TensorRT FP16 engine, BoT-SORT with camera-motion compensation, thresholds tuned on validation) reaches **HOTA 53.7, MOTA 54.2 and IDF1 69.9 with 89% fewer ID switches** than the baseline (640 px detector + ByteTrack: HOTA 41.0, MOTA 33.2, IDF1 48.7), at 49 FPS end to end in Python.

**Detection speed:** on an RTX 4000 Ada at batch 1, the final pipeline (YOLO11s at 1280 px, TensorRT FP16, C++ with a fused CUDA pre-processing kernel) runs at **3.9 ms per frame (257 FPS) with 0.327 mAP50-95**. The naive baseline (YOLO11s at 640 px, PyTorch, Python) takes 4.8 ms for 0.224 mAP50-95. The deployed pipeline is **46% more accurate and also faster** than the baseline.

| Pipeline | Input | mAP50-95 | Latency (ms) | FPS |
|---|---:|---:|---:|---:|
| PyTorch, Python (baseline) | 640 | 0.224 | 4.81 | 208 |
| PyTorch, Python | 1280 | 0.331 | 9.05 | 110 |
| TensorRT FP16, Python | 1280 | 0.327 | 6.88 | 145 |
| **TensorRT FP16, C++ + CUDA pre-processing** | **1280** | **0.327** | **3.89** | **257** |
| TensorRT INT8, C++ + CUDA pre-processing | 1280 | 0.316 | 2.80 | 357 |

Latency is end to end per frame: upload, pre-processing, inference, download and NMS. JPEG decoding is excluded. Each figure is the median of 300 timed runs after 50 warm-up runs on VisDrone validation images. mAP is measured with the Ultralytics validator on the VisDrone-DET validation set (548 images).

## Stage 1: resolution matters most for small objects

Fine-tuning YOLO11s (COCO-pretrained) on VisDrone-DET for 100 epochs:

| Input | mAP50 | mAP50-95 |
|---:|---:|---:|
| 640 | 0.393 | 0.225 |
| 1280 | **0.546** | **0.335** (+49% relative) |

Every class improves, and the smallest objects gain the most in relative terms:

| Class | 640 | 1280 | Gain |
|---|---:|---:|---:|
| pedestrian | 0.186 | 0.327 | +0.141 (+76%) |
| people | 0.115 | 0.213 | +0.098 |
| bicycle | 0.055 | 0.169 | +0.114 (×3.1) |
| car | 0.540 | 0.635 | +0.095 |
| van | 0.310 | 0.404 | +0.095 |
| truck | 0.235 | 0.334 | +0.099 |
| tricycle | 0.152 | 0.265 | +0.113 |
| awning-tricycle | 0.086 | 0.133 | +0.047 |
| bus | 0.376 | 0.547 | +0.171 |
| motor | 0.198 | 0.319 | +0.120 |

The cost is four times as many pixels per frame. Stages 2 and 3 win that back.

## Stage 2: TensorRT export (Python)

Same weights, same validation set, batch 1, measured through the Ultralytics Python API:

| Model | Format | mAP50-95 | Δ | Median ms | Speed-up |
|---|---|---:|---:|---:|---:|
| 640 | PyTorch | 0.2235 | | 4.81 | 1.00× |
| 640 | TensorRT FP16 | 0.2229 | −0.0006 | 3.80 | 1.27× |
| 640 | TensorRT INT8 | 0.2152 | −0.0083 | 3.55 | 1.35× |
| 1280 | PyTorch | 0.3313 | | 9.05 | 1.00× |
| 1280 | TensorRT FP16 | 0.3272 | −0.0041 | 6.88 | 1.32× |
| 1280 | TensorRT INT8 | 0.3156 | −0.0157 | 5.81 | 1.56× |

FP16 is effectively lossless. INT8 (post-training calibration on 25% of the training set) costs 3.7–4.7% mAP. The speed-ups are modest because Python pre- and post-processing dominate at batch 1, which is what Stage 3 addresses. ONNX Runtime results are not reported: in this environment it fell back to CPU execution, so its timings are not comparable.

## Stage 3: C++ engine with a fused CUDA pre-processing kernel

`cpp/` is a standalone C++17 TensorRT runner:
- **One fused CUDA kernel** performs bilinear resize, letterbox padding, BGR→RGB, normalisation and HWC→CHW in a single pass, replacing several OpenCV/NumPy CPU operations.
- **TensorRT `enqueueV3`** on a dedicated stream, with pinned host buffers.
- **Decoding and class-aware NMS** on the CPU.
- **Per-stage timing** with CUDA events.

Median per-stage latency in ms (300 runs):

| Engine | Upload | Pre-proc (CUDA) | Inference | Download | Post-proc | Total | p95 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 640 FP16 | 0.11 | 0.01 | 1.16 | 0.03 | 0.05 | 1.57 | 2.17 | 637 |
| 640 INT8 | 0.12 | 0.01 | 0.88 | 0.03 | 0.05 | 1.27 | 1.87 | 786 |
| 1280 FP16 | 0.12 | 0.04 | 3.19 | 0.12 | 0.16 | 3.91 | 4.50 | 256 |
| 1280 INT8 | 0.11 | 0.04 | 2.09 | 0.11 | 0.17 | 2.80 | 3.39 | 358 |

The custom kernel makes pre-processing essentially free (0.04 ms at 1280), so 82% of the remaining time is the network itself. Moving the same FP16 engine from Python to C++ makes it **1.8× faster** (6.88 → 3.89 ms).

### Correctness: C++ vs Python parity, and an INT8 finding

To check the C++ pipeline, its detections are compared with the Python path running the same engine (100 validation images, conf 0.25, NMS IoU 0.45). Agreement is the fraction of Python detections matched by a same-class C++ box at the given IoU:

| Engine | IoU ≥ 0.5 | IoU ≥ 0.75 | IoU ≥ 0.9 |
|---|---:|---:|---:|
| 640 FP16 | 0.995 | 0.994 | 0.987 |
| 640 INT8 | 0.947 | 0.928 | 0.806 |
| 1280 FP16 | 0.996 | 0.996 | 0.986 |
| 1280 INT8 | 0.938 | 0.923 | 0.740 |

FP16 agrees almost perfectly. INT8 agrees noticeably less, even though both paths run the same engine file with identical thresholds and fixed input shapes. To find out why, each engine was run on every image and on a copy with invisible ±1 pixel-value noise:

| Engine | IoU ≥ 0.5 | IoU ≥ 0.75 | IoU ≥ 0.9 |
|---|---:|---:|---:|
| 640 FP16 | 0.988 | 0.986 | 0.969 |
| 640 INT8 | 0.934 | 0.908 | 0.739 |
| 1280 FP16 | 0.990 | 0.989 | 0.970 |
| 1280 INT8 | 0.938 | 0.923 | 0.733 |

The C++/Python disagreement for INT8 matches the INT8 engine's own sensitivity to imperceptible noise (0.740 vs 0.733 at IoU 0.9). So the C++ pipeline is correct, and the gap is a property of the quantised model. **Post-training INT8 makes this small-object detector fragile**: ±1 intensity noise changes about 6% of its detections outright and shifts the box edges on about 27% of them, against 1% and 3% for FP16. Tiny objects sit right at the resolution limit of the quantised activations. FP16 is therefore the deployment default here. Better INT8 (more calibration data, keeping sensitive layers in FP16, or explicit Q/DQ quantisation) is left for the Jetson stage, where the extra speed matters.

## Stage 4: multi-object tracking on VisDrone2019-MOT

Detections are linked across frames by a tracker, and the result is scored with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, CLEAR/MOTA, Identity/IDF1) on the 7 validation sequences (2,846 frames, 72,690 evaluated boxes). Following the VisDrone-MOT protocol, the five evaluated categories (pedestrian, car, van, truck, bus) are scored class-agnostically, and predictions mostly inside annotated *ignored regions* are removed. Both detectors run as TensorRT FP16 engines.

### Ablation: detector resolution × camera-motion compensation (default thresholds)

| Detector | Tracker | HOTA | DetA | AssA | MOTA | IDF1 | ID switches | FPS (Python) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | ByteTrack | 41.6 | 37.5 | 48.0 | 30.7 | 48.5 | 1,226 | 88.7 |
| 640 | BoT-SORT (CMC) | 47.1 | 38.8 | 59.0 | 31.3 | 56.6 | 702 | 41.1 |
| 1280 | ByteTrack | 44.4 | 39.6 | 51.8 | 27.0 | 52.4 | 1,175 | 65.8 |
| 1280 | BoT-SORT (CMC) | 48.9 | 40.3 | 61.5 | 26.1 | 59.5 | 498 | 36.6 |

- **Camera-motion compensation is the biggest single gain.** The drone's own motion makes every object jump between frames. BoT-SORT's global motion compensation (sparse optical flow) cuts ID switches by 43–58% and raises association accuracy (AssA) by about 10 points, with either detector. It costs speed, because motion is estimated on every full-resolution frame on the CPU.
- **The 1280 detector improves detection (DetA) and identity (IDF1)**, but at default thresholds MOTA *drops*. The breakdown explains why: 1280 finds 22% more true objects and misses 32% fewer (FN 29,310 → 19,863), but adds 68% more false positives (FP 19,819 → 33,342). MOTA weighs every error equally, so the extra false positives dominate. HOTA balances detection and association, so it still improves.

### Tuning the tracker's confidence thresholds (1280 + BoT-SORT)

`track_high_thresh` and `new_track_thresh` are raised together from the Ultralytics default of 0.25:

| Threshold | HOTA | MOTA | IDF1 | ID switches | Det. precision | Det. recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 (default) | 48.9 | 26.1 | 59.5 | 498 | 49.8 | 59.1 |
| 0.35 | 49.8 | 30.0 | 61.0 | 394 | 51.6 | 58.9 |
| 0.45 | 51.8 | 38.4 | 64.9 | 247 | 55.8 | 57.6 |
| 0.55 | 53.1 | 44.1 | 66.8 | 181 | 59.8 | 56.0 |
| **0.65** | **53.2** | **47.3** | **67.0** | **116** | 64.2 | 52.5 |

Against the baseline (640 + ByteTrack, default thresholds), the tuned pipeline improves every metric: **HOTA 41.6 → 53.2, MOTA 30.7 → 47.3, IDF1 48.5 → 67.0, and ID switches −91% (1,226 → 116).**

The threshold was selected on validation, so these validation numbers are optimistic. The selection rule (maximum HOTA → 0.65) was fixed before touching test-dev.

### Confirmation on the held-out test-dev split

VisDrone2019-MOT test-dev (17 sequences, 6,635 frames, 232,378 evaluated boxes) was evaluated **once**, after every choice above was fixed (`scripts/run_stage4_testdev.sh`):

| Detector | Tracker | Thresholds | HOTA | DetA | AssA | MOTA | IDF1 | ID switches | FPS (Python) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | ByteTrack | default | 41.0 | 39.6 | 44.0 | 33.2 | 48.7 | 4,911 | 75.0 |
| 640 | BoT-SORT | default | 45.9 | 41.0 | 53.0 | 34.9 | 56.0 | 3,206 | 55.7 |
| 1280 | ByteTrack | default | 45.1 | 42.4 | 49.4 | 34.6 | 55.1 | 3,677 | 60.0 |
| 1280 | BoT-SORT | default | 49.9 | 43.4 | 59.0 | 35.5 | 62.4 | 1,957 | 47.2 |
| 640 | BoT-SORT | tuned (0.65) | 48.3 | 38.9 | 60.8 | 45.9 | 61.7 | 372 | 58.3 |
| **1280** | **BoT-SORT** | **tuned (0.65)** | **53.7** | **47.3** | **62.3** | **54.2** | **69.9** | **557** | **48.9** |

- **The validation conclusions hold.** The final configuration scores HOTA 53.7 on test-dev, against 53.2 on validation, so tuning on validation did not overfit.
- **The gain splits into roughly equal thirds.** Of the +12.7 HOTA over the baseline, camera-motion compensation gives +4.8, the 1280 detector +4.1, and threshold tuning +3.8. The control run confirms the detector still matters after tuning: 640 at the same tuned threshold reaches 48.3 HOTA, against 53.7 for 1280.
- **Fewer ID switches can also mean fewer objects tracked.** The tuned 640 run has the fewest ID switches (372), but only because its recall collapses to 42.6% (55.5% for 1280). ID-switch counts should always be read next to recall.
- **The validation MOTA dip does not replicate.** On test-dev the 1280 detector's MOTA is slightly *higher* than 640's at default thresholds (34.6 vs 33.2). The extra false positives seen on validation were partly specific to that split.

## Plan

| Stage | What | Status |
|---|---|---|
| 1 | Fine-tune YOLO11s on VisDrone-DET at 640 vs 1280 | ✅ |
| 2 | Export to ONNX / TensorRT FP16 / INT8; accuracy vs latency | ✅ |
| 3 | C++ engine, fused CUDA pre-processing, per-stage timing, parity | ✅ |
| 4 | Tracking on VisDrone-MOT: detector × camera-motion ablation, threshold tuning on val, one-shot test-dev | ✅ |
| 5 | Multi-threaded pipeline (decode, infer, track in parallel) on real video | planned |
| Stretch | Jetson Orin deployment with improved INT8; camera calibration | planned |

## Reproduce

Tested on an NVIDIA RTX 4000 Ada (20 GB), CUDA 12.9, TensorRT 11.4, Python 3.12, Ubuntu 24.04.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

bash scripts/run_stage1_training.sh                    # stage 1: 640 and 1280 (resumable)
python scripts/export_and_benchmark.py --weights runs/detect/yolo11s_1280/weights/best.pt --imgsz 1280   # stage 2
bash scripts/run_stage3.sh                             # stage 3: C++ benchmarks + parity (GPU must be idle)
bash scripts/run_int8_diagnosis.sh                     # multi-threshold parity + noise sensitivity
bash scripts/run_stage4.sh                             # stage 4: tracking ablation on VisDrone-MOT val
bash scripts/run_stage4_sweep.sh                       # stage 4b: tracker threshold sweep (val)
bash scripts/run_stage4_testdev.sh                     # stage 4c: one-shot test-dev confirmation
```

VisDrone-DET (about 2.3 GB) downloads automatically on the first training run.

### Building the C++ engine

Needs CUDA 12.x, OpenCV 4 and the TensorRT headers of the **same version** as the Python TensorRT package that built the engines (`git clone --branch v11.4 --sparse https://github.com/NVIDIA/TensorRT`, then `git sparse-checkout set include`). Use the compiler OpenCV was built with (on Ubuntu 24.04, the system GCC 13):

```bash
cmake -S cpp -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DTENSORRT_INCLUDE_DIR=<TensorRT>/include \
  -DTENSORRT_LIB_DIR=<venv>/lib/python3.12/site-packages/tensorrt_libs
cmake --build build -j
```

## Data

[VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) (Zhu et al., TPAMI 2021). Detection: 6,471 training and 548 validation images, 10 classes. Tracking: VisDrone2019-MOT validation (7 sequences) for all development, test-dev (17 sequences) for the final confirmation only.
