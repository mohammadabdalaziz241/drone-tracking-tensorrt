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
  return decode_and_nms(h_output_, num_anchors_, num_classes_, lb, img_w, img_h, conf_thres_,
                        iou_thres_, max_det_);
}
