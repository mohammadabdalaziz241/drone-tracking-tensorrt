#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

#include "letterbox.h"  // Letterbox, compute_letterbox (plain C++)

// One fused kernel: bilinear resize + letterbox pad + BGR->RGB + /255 + HWC->CHW.
// src: device uint8 BGR image (src_pitch bytes per row); dst: device float32 [3, dst_h, dst_w].
void launch_letterbox_preprocess(const uint8_t* src, int src_w, int src_h, int src_pitch,
                                 float* dst, int dst_w, int dst_h, const Letterbox& lb,
                                 cudaStream_t stream);
