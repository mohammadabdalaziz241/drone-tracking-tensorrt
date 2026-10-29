#include "postprocess.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>

size_t engine_payload_offset(const std::vector<char>& blob) {
  if (blob.size() < 8) return 0;
  int32_t len = 0;
  std::memcpy(&len, blob.data(), 4);
  if (len > 0 && static_cast<size_t>(len) + 4 < blob.size() && blob[4] == '{') return 4 + len;
  return 0;
}

float box_iou(const Detection& a, const Detection& b) {
  const float ix = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float iy = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = ix * iy;
  const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

std::vector<Detection> decode_and_nms(const float* out, int num_anchors, int num_classes,
                                      const Letterbox& lb, int img_w, int img_h,
                                      float conf_thres, float iou_thres, int max_det) {
  // Output layout: channel-major [4 + num_classes][num_anchors]; boxes are (cx, cy, w, h)
  // in letterboxed input pixels.
  const int n = num_anchors;
  std::vector<Detection> cand;
  cand.reserve(1024);
  for (int i = 0; i < n; ++i) {
    int best_c = 0;
    float best = out[4 * n + i];
    for (int c = 1; c < num_classes; ++c) {
      const float s = out[(4 + c) * n + i];
      if (s > best) { best = s; best_c = c; }
    }
    if (best < conf_thres) continue;
    const float cx = out[i], cy = out[n + i], w = out[2 * n + i], h = out[3 * n + i];
    Detection d;
    d.x1 = std::clamp((cx - 0.5f * w - lb.left) / lb.gain, 0.f, static_cast<float>(img_w));
    d.y1 = std::clamp((cy - 0.5f * h - lb.top) / lb.gain, 0.f, static_cast<float>(img_h));
    d.x2 = std::clamp((cx + 0.5f * w - lb.left) / lb.gain, 0.f, static_cast<float>(img_w));
    d.y2 = std::clamp((cy + 0.5f * h - lb.top) / lb.gain, 0.f, static_cast<float>(img_h));
    d.score = best;
    d.cls = best_c;
    cand.push_back(d);
  }

  // Class-aware greedy NMS.
  std::vector<int> order(cand.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return cand[a].score > cand[b].score; });
  std::vector<char> removed(cand.size(), 0);
  std::vector<Detection> keep;
  keep.reserve(std::min<size_t>(cand.size(), max_det));
  for (size_t oi = 0; oi < order.size() && static_cast<int>(keep.size()) < max_det; ++oi) {
    const int i = order[oi];
    if (removed[i]) continue;
    keep.push_back(cand[i]);
    for (size_t oj = oi + 1; oj < order.size(); ++oj) {
      const int j = order[oj];
      if (!removed[j] && cand[j].cls == cand[i].cls && box_iou(cand[i], cand[j]) > iou_thres) removed[j] = 1;
    }
  }
  return keep;
}
