// trt_bench: C++ TensorRT detector benchmark with per-stage latency breakdown.
//
//   trt_bench --engine best_trt_fp16.engine --images <dir> [--runs 300] [--warmup 50]
//             [--max-images 100] [--conf 0.25] [--iou 0.45]
//             [--out results.json] [--dump detections.json] [--vis out_dir]
//
// Images are decoded into memory before timing, so JPEG decoding is not counted.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "detector.h"

namespace fs = std::filesystem;

namespace {

std::map<std::string, std::string> parse_args(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + k);
    a[k.substr(2)] = (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) ? argv[++i] : "1";
  }
  return a;
}

std::string get(const std::map<std::string, std::string>& a, const std::string& k, const std::string& def) {
  auto it = a.find(k);
  return it == a.end() ? def : it->second;
}

float percentile(std::vector<float> v, float p) {
  if (v.empty()) return 0.f;
  std::sort(v.begin(), v.end());
  const size_t idx = std::min(v.size() - 1, static_cast<size_t>(p / 100.f * (v.size() - 1) + 0.5f));
  return v[idx];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);
    const std::string engine_path = get(args, "engine", "");
    const std::string image_dir = get(args, "images", "");
    if (engine_path.empty() || image_dir.empty()) {
      std::cerr << "usage: trt_bench --engine <file.engine> --images <dir> [options]\n";
      return 2;
    }
    const int runs = std::stoi(get(args, "runs", "300"));
    const int warmup = std::stoi(get(args, "warmup", "50"));
    const size_t max_images = std::stoul(get(args, "max-images", "100"));
    const float conf = std::stof(get(args, "conf", "0.25"));
    const float iou = std::stof(get(args, "iou", "0.45"));

    std::vector<fs::path> paths;
    for (const auto& e : fs::directory_iterator(image_dir)) {
      const auto ext = e.path().extension().string();
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") paths.push_back(e.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.size() > max_images) paths.resize(max_images);
    std::vector<cv::Mat> images;
    for (const auto& p : paths) images.push_back(cv::imread(p.string(), cv::IMREAD_COLOR));
    if (images.empty()) throw std::runtime_error("no images in " + image_dir);

    TrtDetector det(engine_path, conf, iou);
    std::cerr << "engine input " << det.input_width() << "x" << det.input_height() << ", "
              << det.num_classes() << " classes, " << det.num_anchors() << " anchors; "
              << images.size() << " images loaded\n";

    for (int i = 0; i < warmup; ++i) det.detect(images[i % images.size()]);

    std::vector<float> up, pre, inf, down, post, total;
    size_t n_dets = 0;
    for (int i = 0; i < runs; ++i) {
      StageTimes t;
      n_dets += det.detect(images[i % images.size()], &t).size();
      up.push_back(t.upload);
      pre.push_back(t.preprocess);
      inf.push_back(t.inference);
      down.push_back(t.download);
      post.push_back(t.postprocess);
      total.push_back(t.total);
    }

    std::ostringstream js;
    js << std::fixed << std::setprecision(3) << "{\n"
       << "  \"engine\": \"" << engine_path << "\",\n"
       << "  \"input\": [" << det.input_width() << ", " << det.input_height() << "],\n"
       << "  \"runs\": " << runs << ", \"warmup\": " << warmup << ", \"images\": " << images.size() << ",\n"
       << "  \"conf\": " << conf << ", \"iou\": " << iou << ",\n"
       << "  \"median_ms\": {\"upload\": " << percentile(up, 50) << ", \"preprocess\": " << percentile(pre, 50)
       << ", \"inference\": " << percentile(inf, 50) << ", \"download\": " << percentile(down, 50)
       << ", \"postprocess\": " << percentile(post, 50) << ", \"total\": " << percentile(total, 50) << "},\n"
       << "  \"p95_total_ms\": " << percentile(total, 95) << ",\n"
       << "  \"fps_batch1\": " << 1000.f / percentile(total, 50) << ",\n"
       << "  \"mean_detections_per_image\": " << static_cast<float>(n_dets) / runs << "\n}\n";
    std::cout << js.str();
    if (args.count("out")) std::ofstream(get(args, "out", "")) << js.str();

    if (args.count("dump")) {  // detections for the Python parity check
      std::ofstream out(get(args, "dump", ""));
      out << std::setprecision(6) << "{";
      for (size_t i = 0; i < images.size(); ++i) {
        out << (i ? "," : "") << "\n\"" << paths[i].filename().string() << "\": [";
        const auto dets = det.detect(images[i]);
        for (size_t j = 0; j < dets.size(); ++j) {
          const auto& d = dets[j];
          out << (j ? "," : "") << "[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << ","
              << d.score << "," << d.cls << "]";
        }
        out << "]";
      }
      out << "\n}\n";
    }

    if (args.count("vis")) {
      const fs::path vis_dir = get(args, "vis", "");
      fs::create_directories(vis_dir);
      for (size_t i = 0; i < std::min<size_t>(5, images.size()); ++i) {
        cv::Mat canvas = images[i].clone();
        for (const auto& d : det.detect(images[i])) {
          const cv::Scalar colour(37 * (d.cls + 1) % 255, 97 * (d.cls + 1) % 255, 157 * (d.cls + 1) % 255);
          cv::rectangle(canvas, cv::Point2f(d.x1, d.y1), cv::Point2f(d.x2, d.y2), colour, 2);
        }
        cv::imwrite((vis_dir / paths[i].filename()).string(), canvas);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
