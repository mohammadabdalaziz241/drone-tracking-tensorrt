#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

// Letterbox geometry, matching Ultralytics' LetterBox (centred padding, value 114).
struct Letterbox {
  float gain;   // resize ratio applied to the source image
  int new_w;    // resized (unpadded) width
  int new_h;    // resized (unpadded) height
  int left;     // horizontal padding before the image
  int top;      // vertical padding before the image
};

Letterbox compute_letterbox(int src_w, int src_h, int dst_w, int dst_h);

// One fused kernel: bilinear resize + letterbox pad + BGR->RGB + /255 + HWC->CHW.
// src: device uint8 BGR image (src_pitch bytes per row); dst: device float32 [3, dst_h, dst_w].
void launch_letterbox_preprocess(const uint8_t* src, int src_w, int src_h, int src_pitch,
                                 float* dst, int dst_w, int dst_h, const Letterbox& lb,
                                 cudaStream_t stream);
