#pragma once

// Decode + NMS and engine-file parsing. Plain C++ (no CUDA or TensorRT headers) so it can be
// unit-tested on a CPU-only machine; TrtDetector calls these on its host output buffer.

#include <cstddef>
#include <vector>

#include "letterbox.h"

struct Detection {
  float x1, y1, x2, y2;  // box in original-image pixels
  float score;
  int cls;
};

// Decode a YOLO head output and apply class-aware greedy NMS.
// out: channel-major [4 + num_classes][num_anchors]; boxes are (cx, cy, w, h) in letterboxed
// input pixels. Returns at most max_det detections, highest score first, mapped back to the
// original image and clamped to [0, img_w] x [0, img_h].
std::vector<Detection> decode_and_nms(const float* out, int num_anchors, int num_classes,
                                      const Letterbox& lb, int img_w, int img_h,
                                      float conf_thres, float iou_thres, int max_det);

// Intersection over union of two boxes; 0 when the union is empty.
float box_iou(const Detection& a, const Detection& b);

// Ultralytics prepends a 4-byte little-endian length + JSON metadata to its .engine files.
// Returns the byte offset of the serialized TensorRT engine (0 for a plain engine).
size_t engine_payload_offset(const std::vector<char>& blob);
