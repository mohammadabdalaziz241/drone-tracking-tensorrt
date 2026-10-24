#include "preprocess.cuh"

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

__global__ void letterbox_kernel(const uint8_t* __restrict__ src, int src_w, int src_h, int src_pitch,
                                 float* __restrict__ dst, int dst_w, int dst_h,
                                 int new_w, int new_h, int left, int top,
                                 float scale_x, float scale_y) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;

  float b = 114.f, g = 114.f, r = 114.f;
  const int ux = x - left;
  const int uy = y - top;
  if (ux >= 0 && ux < new_w && uy >= 0 && uy < new_h) {
    // Half-pixel-centre mapping, as in cv::resize(INTER_LINEAR).
    float fx = fmaxf((ux + 0.5f) * scale_x - 0.5f, 0.f);
    float fy = fmaxf((uy + 0.5f) * scale_y - 0.5f, 0.f);
    int x0 = min(static_cast<int>(fx), src_w - 1);
    int y0 = min(static_cast<int>(fy), src_h - 1);
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);
    const float ax = fx - x0, ay = fy - y0;
    const float w00 = (1.f - ax) * (1.f - ay), w01 = ax * (1.f - ay);
    const float w10 = (1.f - ax) * ay, w11 = ax * ay;
    const uint8_t* p00 = src + y0 * src_pitch + x0 * 3;
    const uint8_t* p01 = src + y0 * src_pitch + x1 * 3;
    const uint8_t* p10 = src + y1 * src_pitch + x0 * 3;
    const uint8_t* p11 = src + y1 * src_pitch + x1 * 3;
    // cv::resize on 8-bit images rounds the interpolated value back to uint8; do the same so
    // the network sees bit-identical inputs to the Python/OpenCV path.
    b = rintf(w00 * p00[0] + w01 * p01[0] + w10 * p10[0] + w11 * p11[0]);
    g = rintf(w00 * p00[1] + w01 * p01[1] + w10 * p10[1] + w11 * p11[1]);
    r = rintf(w00 * p00[2] + w01 * p01[2] + w10 * p10[2] + w11 * p11[2]);
  }
  const int plane = dst_w * dst_h;
  const int idx = y * dst_w + x;
  constexpr float inv255 = 1.f / 255.f;
  dst[idx] = r * inv255;              // channel 0 = R
  dst[plane + idx] = g * inv255;      // channel 1 = G
  dst[2 * plane + idx] = b * inv255;  // channel 2 = B
}

void launch_letterbox_preprocess(const uint8_t* src, int src_w, int src_h, int src_pitch,
                                 float* dst, int dst_w, int dst_h, const Letterbox& lb,
                                 cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  const float scale_x = static_cast<float>(src_w) / lb.new_w;
  const float scale_y = static_cast<float>(src_h) / lb.new_h;
  letterbox_kernel<<<grid, block, 0, stream>>>(src, src_w, src_h, src_pitch, dst, dst_w, dst_h,
                                               lb.new_w, lb.new_h, lb.left, lb.top, scale_x, scale_y);
}
