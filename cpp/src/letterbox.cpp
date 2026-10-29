#include "letterbox.h"

#include <algorithm>
#include <cmath>

Letterbox compute_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
  Letterbox lb{};
  lb.gain = std::min(static_cast<float>(dst_w) / src_w, static_cast<float>(dst_h) / src_h);
  lb.new_w = static_cast<int>(std::lround(src_w * lb.gain));
  lb.new_h = static_cast<int>(std::lround(src_h * lb.gain));
  const float dw = (dst_w - lb.new_w) / 2.0f;
  const float dh = (dst_h - lb.new_h) / 2.0f;
  lb.left = static_cast<int>(std::lround(dw - 0.1f));  // same rounding as Ultralytics
  lb.top = static_cast<int>(std::lround(dh - 0.1f));
  return lb;
}
