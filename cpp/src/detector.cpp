#include "detector.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iterator>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

#define CUDA_CHECK(call)                                                                  \
  do {                                                                                    \
    cudaError_t err__ = (call);                                                           \
    if (err__ != cudaSuccess)                                                             \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));      \
  } while (0)

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << "\n";
  }
};
Logger g_logger;

// Ultralytics prepends a 4-byte little-endian length + JSON metadata to its .engine files.
size_t engine_payload_offset(const std::vector<char>& blob) {
  if (blob.size() < 8) return 0;
  int32_t len = 0;
  std::memcpy(&len, blob.data(), 4);
  if (len > 0 && static_cast<size_t>(len) + 4 < blob.size() && blob[4] == '{') return 4 + len;
  return 0;
}

float iou(const Detection& a, const Detection& b) {
  const float ix = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float iy = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = ix * iy;
  const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

}  // namespace

TrtDetector::TrtDetector(const std::string& engine_path, float conf_thres, float iou_thres, int max_det)
    : conf_thres_(conf_thres), iou_thres_(iou_thres), max_det_(max_det) {
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open engine: " + engine_path);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const size_t off = engine_payload_offset(blob);

  runtime_.reset(nvinfer1::createInferRuntime(g_logger));
  engine_.reset(runtime_->deserializeCudaEngine(blob.data() + off, blob.size() - off));
  if (!engine_) throw std::runtime_error("failed to deserialize engine (TensorRT version mismatch?)");
  context_.reset(engine_->createExecutionContext());

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getIOTensorName(i);
    const auto dims = engine_->getTensorShape(name);
    if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
      throw std::runtime_error(std::string("expected float32 I/O tensor: ") + name);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      in_name_ = name;  // [1, 3, H, W]
      in_h_ = static_cast<int>(dims.d[2]);
      in_w_ = static_cast<int>(dims.d[3]);
    } else {
      out_name_ = name;  // [1, 4 + num_classes, num_anchors]
      out_channels_ = static_cast<int>(dims.d[1]);
      num_anchors_ = static_cast<int>(dims.d[2]);
      num_classes_ = out_channels_ - 4;
    }
  }
  if (in_name_.empty() || out_name_.empty() || num_classes_ <= 0)
    throw std::runtime_error("unexpected engine layout");

  CUDA_CHECK(cudaStreamCreate(&stream_));
  for (auto& e : ev_) CUDA_CHECK(cudaEventCreate(&e));
  const size_t out_elems = static_cast<size_t>(out_channels_) * num_anchors_;
  CUDA_CHECK(cudaMalloc(&d_input_, sizeof(float) * 3 * in_w_ * in_h_));
  CUDA_CHECK(cudaMalloc(&d_output_, sizeof(float) * out_elems));
  CUDA_CHECK(cudaMallocHost(&h_output_, sizeof(float) * out_elems));
  context_->setTensorAddress(in_name_.c_str(), d_input_);
  context_->setTensorAddress(out_name_.c_str(), d_output_);
}

TrtDetector::~TrtDetector() {
  context_.reset();
  engine_.reset();
  runtime_.reset();
  cudaFree(d_input_);
  cudaFree(d_output_);
  cudaFree(d_image_);
  cudaFreeHost(h_output_);
  cudaFreeHost(h_image_);
  for (auto& e : ev_) cudaEventDestroy(e);
  if (stream_) cudaStreamDestroy(stream_);
}

void TrtDetector::ensure_image_capacity(size_t bytes) {
  if (bytes <= image_capacity_) return;
  cudaFree(d_image_);
  cudaFreeHost(h_image_);
  CUDA_CHECK(cudaMalloc(&d_image_, bytes));
  CUDA_CHECK(cudaMallocHost(&h_image_, bytes));
  image_capacity_ = bytes;
}

std::vector<Detection> TrtDetector::detect(const cv::Mat& bgr, StageTimes* times) {
  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();
  if (bgr.empty() || bgr.type() != CV_8UC3) throw std::runtime_error("expected a CV_8UC3 BGR image");

  const cv::Mat img = bgr.isContinuous() ? bgr : bgr.clone();
  const size_t bytes = img.total() * img.elemSize();
  ensure_image_capacity(bytes);
  std::memcpy(h_image_, img.data, bytes);
  const Letterbox lb = compute_letterbox(img.cols, img.rows, in_w_, in_h_);
  const size_t out_bytes = sizeof(float) * out_channels_ * num_anchors_;

  CUDA_CHECK(cudaEventRecord(ev_[0], stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_image_, h_image_, bytes, cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaEventRecord(ev_[1], stream_));
  launch_letterbox_preprocess(d_image_, img.cols, img.rows, static_cast<int>(img.step[0]), d_input_,
                              in_w_, in_h_, lb, stream_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev_[2], stream_));
  if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
  CUDA_CHECK(cudaEventRecord(ev_[3], stream_));
  CUDA_CHECK(cudaMemcpyAsync(h_output_, d_output_, out_bytes, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaEventRecord(ev_[4], stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  const auto t_post = clock::now();
  auto dets = postprocess(lb, img.cols, img.rows);
  const auto t_end = clock::now();

  if (times) {
    CUDA_CHECK(cudaEventElapsedTime(&times->upload, ev_[0], ev_[1]));
    CUDA_CHECK(cudaEventElapsedTime(&times->preprocess, ev_[1], ev_[2]));
    CUDA_CHECK(cudaEventElapsedTime(&times->inference, ev_[2], ev_[3]));
    CUDA_CHECK(cudaEventElapsedTime(&times->download, ev_[3], ev_[4]));
    times->postprocess = std::chrono::duration<float, std::milli>(t_end - t_post).count();
    times->total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
  }
  return dets;
}

std::vector<Detection> TrtDetector::postprocess(const Letterbox& lb, int img_w, int img_h) const {
  // Output layout: channel-major [4 + num_classes][num_anchors]; boxes are (cx, cy, w, h)
  // in letterboxed input pixels.
  const float* out = h_output_;
  const int n = num_anchors_;
  std::vector<Detection> cand;
  cand.reserve(1024);
  for (int i = 0; i < n; ++i) {
    int best_c = 0;
    float best = out[4 * n + i];
    for (int c = 1; c < num_classes_; ++c) {
      const float s = out[(4 + c) * n + i];
      if (s > best) { best = s; best_c = c; }
    }
    if (best < conf_thres_) continue;
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
  keep.reserve(std::min<size_t>(cand.size(), max_det_));
  for (size_t oi = 0; oi < order.size() && static_cast<int>(keep.size()) < max_det_; ++oi) {
    const int i = order[oi];
    if (removed[i]) continue;
    keep.push_back(cand[i]);
    for (size_t oj = oi + 1; oj < order.size(); ++oj) {
      const int j = order[oj];
      if (!removed[j] && cand[j].cls == cand[i].cls && iou(cand[i], cand[j]) > iou_thres_) removed[j] = 1;
    }
  }
  return keep;
}
