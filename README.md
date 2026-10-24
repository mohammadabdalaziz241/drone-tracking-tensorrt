# Drone small-object detection and tracking with TensorRT

> **Status: in progress.** Results are added here as each stage is completed and measured.

Detecting and tracking small objects (pedestrians, vehicles) in aerial drone video is hard: objects are often under 32×32 pixels, crowded, and seen from unusual angles. This project builds a full pipeline for it, from fine-tuning a detector to a deployed C++/CUDA TensorRT tracker, and measures what every optimisation step gains and costs.

## Plan

| Stage | What | Measured |
|---|---|---|
| 1 | Fine-tune YOLO11s on VisDrone-DET at 640 vs 1280 input | mAP50-95, per-class and small-object AP |
| 2 | Export to ONNX and TensorRT (FP32, FP16, INT8 with calibration) | accuracy drop vs latency and throughput |
| 3 | C++ inference engine with custom CUDA pre-processing and NMS | end-to-end FPS, per-stage latency |
| 4 | ByteTrack multi-object tracking on VisDrone-MOT | HOTA, IDF1, MOTA, ID switches |
| 5 | Multi-threaded pipeline (decode, infer, track in parallel) | sustained FPS on real video |
| Stretch | Jetson Orin deployment; camera calibration for real-world units | edge FPS and power |

## Setup

Tested on an NVIDIA RTX 4000 Ada (20 GB), CUDA 12.9, Python 3.12.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

VisDrone-DET (about 2.3 GB) downloads automatically on first training run.

### Building the C++ engine

Needs CUDA 12.x, OpenCV 4 and the TensorRT headers of the **same version** as the Python
TensorRT package that built the engines (headers: `git clone --branch v11.4 --sparse https://github.com/NVIDIA/TensorRT`, then `sparse-checkout set include`).
Use the compiler OpenCV was built with (on Ubuntu 24.04, the system GCC 13):

```bash
cmake -S cpp -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DTENSORRT_INCLUDE_DIR=<TensorRT>/include \
  -DTENSORRT_LIB_DIR=<venv>/lib/python3.12/site-packages/tensorrt_libs
cmake --build build -j
```

## Stage 1: detector fine-tuning

```bash
bash scripts/run_stage1_training.sh     # 640 baseline, then 1280 small-object model
```

Each run writes a JSON summary to `results/`.

## Data

[VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) (Zhu et al., TPAMI 2021): 6,471 training and 548 validation images, 10 classes.
