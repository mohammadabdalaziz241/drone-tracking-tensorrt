#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>  // C++ overloads, e.g. cudaMalloc(T**, size_t)

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "postprocess.h"  // Detection, decode_and_nms
#include "preprocess.cuh"

// Per-stage latency of one detect() call, in milliseconds.
struct StageTimes {
  float upload = 0;       // host -> device copy of the raw image (GPU timeline)
  float preprocess = 0;   // fused CUDA letterbox kernel (GPU timeline)
  float inference = 0;    // TensorRT enqueueV3 (GPU timeline)
  float download = 0;     // device -> host copy of the raw output (GPU timeline)
  float postprocess = 0;  // decode + NMS on CPU (wall clock)
  float total = 0;        // whole detect() call (wall clock)
};

class TrtDetector {
 public:
  explicit TrtDetector(const std::string& engine_path, float conf_thres = 0.25f,
                       float iou_thres = 0.45f, int max_det = 300);
  ~TrtDetector();
  TrtDetector(const TrtDetector&) = delete;
  TrtDetector& operator=(const TrtDetector&) = delete;

  std::vector<Detection> detect(const cv::Mat& bgr, StageTimes* times = nullptr);

  int input_width() const { return in_w_; }
  int input_height() const { return in_h_; }
  int num_classes() const { return num_classes_; }
  int num_anchors() const { return num_anchors_; }

 private:
  void ensure_image_capacity(size_t bytes);
  std::vector<Detection> postprocess(const Letterbox& lb, int img_w, int img_h) const;

  float conf_thres_, iou_thres_;
  int max_det_;
  int in_w_ = 0, in_h_ = 0, num_classes_ = 0, num_anchors_ = 0, out_channels_ = 0;
  std::string in_name_, out_name_;

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t ev_[5] = {};
  float* d_input_ = nullptr;
  float* d_output_ = nullptr;
  float* h_output_ = nullptr;  // pinned
  uint8_t* d_image_ = nullptr;
  uint8_t* h_image_ = nullptr;  // pinned staging buffer for uploads
  size_t image_capacity_ = 0;
};
