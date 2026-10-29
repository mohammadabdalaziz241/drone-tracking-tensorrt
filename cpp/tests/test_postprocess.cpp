// Unit tests for decode + class-aware NMS and engine-file parsing (no GPU needed).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "postprocess.h"

namespace {

// A synthetic YOLO head output: channel-major [4 + num_classes][num_anchors].
class HeadOutput {
 public:
  HeadOutput(int num_anchors, int num_classes)
      : n_(num_anchors), c_(num_classes), data_((4 + num_classes) * num_anchors, 0.f) {}

  // Set anchor i to box (cx, cy, w, h) in letterboxed pixels with one class score.
  void set(int i, float cx, float cy, float w, float h, int cls, float score) {
    data_[0 * n_ + i] = cx;
    data_[1 * n_ + i] = cy;
    data_[2 * n_ + i] = w;
    data_[3 * n_ + i] = h;
    data_[(4 + cls) * n_ + i] = score;
  }
  void set_score(int i, int cls, float score) { data_[(4 + cls) * n_ + i] = score; }

  const float* data() const { return data_.data(); }
  int anchors() const { return n_; }
  int classes() const { return c_; }

 private:
  int n_, c_;
  std::vector<float> data_;
};

// Identity letterbox: network pixels equal image pixels.
constexpr Letterbox kIdentity{1.f, 640, 640, 0, 0};

std::vector<Detection> run(const HeadOutput& o, const Letterbox& lb = kIdentity, int img_w = 640,
                           int img_h = 640, float conf = 0.25f, float iou = 0.45f, int max_det = 300) {
  return decode_and_nms(o.data(), o.anchors(), o.classes(), lb, img_w, img_h, conf, iou, max_det);
}

// ---------------------------------------------------------------- box_iou

TEST(BoxIou, IdenticalBoxesIsOne) {
  const Detection a{10, 10, 20, 20, 1.f, 0};
  EXPECT_FLOAT_EQ(box_iou(a, a), 1.f);
}

TEST(BoxIou, DisjointBoxesIsZero) {
  EXPECT_FLOAT_EQ(box_iou({0, 0, 10, 10, 1, 0}, {20, 20, 30, 30, 1, 0}), 0.f);
}

TEST(BoxIou, HalfOverlap) {
  // Two 2x2 boxes sharing a 1x2 strip: intersection 2, union 6.
  EXPECT_FLOAT_EQ(box_iou({0, 0, 2, 2, 1, 0}, {1, 0, 3, 2, 1, 0}), 1.f / 3.f);
}

TEST(BoxIou, ZeroAreaBoxesIsZeroNotNaN) {
  EXPECT_FLOAT_EQ(box_iou({5, 5, 5, 5, 1, 0}, {5, 5, 5, 5, 1, 0}), 0.f);
}

// ---------------------------------------------------------------- decode

TEST(Decode, EmptyWhenNothingPassesThreshold) {
  HeadOutput o(8, 3);
  o.set(0, 100, 100, 20, 20, 1, 0.2f);
  EXPECT_TRUE(run(o).empty());
}

TEST(Decode, ScoreEqualToThresholdIsKept) {
  HeadOutput o(4, 2);
  o.set(0, 100, 100, 20, 20, 0, 0.25f);
  EXPECT_EQ(run(o).size(), 1u);
}

TEST(Decode, PicksTheHighestScoringClass) {
  HeadOutput o(4, 3);
  o.set(0, 100, 100, 20, 20, 0, 0.30f);
  o.set_score(0, 1, 0.70f);
  o.set_score(0, 2, 0.50f);
  const auto d = run(o);
  ASSERT_EQ(d.size(), 1u);
  EXPECT_EQ(d[0].cls, 1);
  EXPECT_FLOAT_EQ(d[0].score, 0.70f);
}

TEST(Decode, ConvertsCentreSizeToCorners) {
  HeadOutput o(4, 1);
  o.set(2, 100, 50, 20, 10, 0, 0.9f);
  const auto d = run(o);
  ASSERT_EQ(d.size(), 1u);
  EXPECT_FLOAT_EQ(d[0].x1, 90.f);
  EXPECT_FLOAT_EQ(d[0].y1, 45.f);
  EXPECT_FLOAT_EQ(d[0].x2, 110.f);
  EXPECT_FLOAT_EQ(d[0].y2, 55.f);
}

TEST(Decode, UndoesLetterboxPaddingAndScale) {
  // 1920x1080 image in a 1280x1280 input: gain 2/3, 280 px padding at the top.
  const Letterbox lb = compute_letterbox(1920, 1080, 1280, 1280);
  ASSERT_EQ(lb.top, 280);
  HeadOutput o(4, 1);
  o.set(0, 640, 640, 64, 32, 0, 0.9f);  // centre of the network input
  const auto d = run(o, lb, 1920, 1080);
  ASSERT_EQ(d.size(), 1u);
  EXPECT_NEAR(d[0].x1, (640 - 32) * 1.5f, 1e-3);
  EXPECT_NEAR(d[0].x2, (640 + 32) * 1.5f, 1e-3);
  EXPECT_NEAR(d[0].y1, (640 - 16 - 280) * 1.5f, 1e-3);
  EXPECT_NEAR(d[0].y2, (640 + 16 - 280) * 1.5f, 1e-3);
}

TEST(Decode, ClampsBoxesToTheImage) {
  HeadOutput o(4, 1);
  o.set(0, 5, 635, 20, 20, 0, 0.9f);  // sticks out on the left and bottom
  const auto d = run(o);
  ASSERT_EQ(d.size(), 1u);
  EXPECT_FLOAT_EQ(d[0].x1, 0.f);
  EXPECT_FLOAT_EQ(d[0].y2, 640.f);
}

// ---------------------------------------------------------------- NMS

TEST(Nms, SuppressesOverlappingBoxOfTheSameClass) {
  HeadOutput o(4, 2);
  o.set(0, 100, 100, 40, 40, 0, 0.9f);
  o.set(1, 102, 100, 40, 40, 0, 0.8f);  // IoU ~0.9 with anchor 0
  const auto d = run(o);
  ASSERT_EQ(d.size(), 1u);
  EXPECT_FLOAT_EQ(d[0].score, 0.9f);
}

TEST(Nms, KeepsOverlappingBoxesOfDifferentClasses) {
  HeadOutput o(4, 2);
  o.set(0, 100, 100, 40, 40, 0, 0.9f);
  o.set(1, 100, 100, 40, 40, 1, 0.8f);  // same box, other class
  EXPECT_EQ(run(o).size(), 2u);
}

TEST(Nms, IouExactlyAtThresholdIsKept) {
  // Boxes (0,0,2,2) and (1,0,3,2): IoU exactly 1/3. Suppression needs IoU > threshold.
  HeadOutput o(4, 1);
  o.set(0, 1, 1, 2, 2, 0, 0.9f);
  o.set(1, 2, 1, 2, 2, 0, 0.8f);
  EXPECT_EQ(run(o, kIdentity, 640, 640, 0.25f, 1.f / 3.f).size(), 2u);
  EXPECT_EQ(run(o, kIdentity, 640, 640, 0.25f, 0.30f).size(), 1u);
}

TEST(Nms, OutputIsSortedByScore) {
  HeadOutput o(6, 1);
  o.set(0, 50, 50, 10, 10, 0, 0.4f);
  o.set(1, 150, 50, 10, 10, 0, 0.9f);
  o.set(2, 250, 50, 10, 10, 0, 0.6f);
  const auto d = run(o);
  ASSERT_EQ(d.size(), 3u);
  EXPECT_FLOAT_EQ(d[0].score, 0.9f);
  EXPECT_FLOAT_EQ(d[1].score, 0.6f);
  EXPECT_FLOAT_EQ(d[2].score, 0.4f);
}

TEST(Nms, RespectsMaxDetections) {
  HeadOutput o(10, 1);
  for (int i = 0; i < 10; ++i) o.set(i, 30.f + 60.f * i, 50, 10, 10, 0, 0.5f + 0.01f * i);
  const auto d = run(o, kIdentity, 640, 640, 0.25f, 0.45f, 3);
  ASSERT_EQ(d.size(), 3u);
  EXPECT_FLOAT_EQ(d[0].score, 0.59f);  // the three best survive
  EXPECT_FLOAT_EQ(d[2].score, 0.57f);
}

TEST(Nms, SuppressedBoxDoesNotSuppressOthers) {
  // A (0.9) suppresses B (0.8); B overlaps C (0.7) but A does not, so C must survive.
  HeadOutput o(4, 1);
  o.set(0, 100, 100, 40, 40, 0, 0.9f);
  o.set(1, 120, 100, 40, 40, 0, 0.8f);  // IoU with A = 1/3 -> suppressed at 0.3
  o.set(2, 140, 100, 40, 40, 0, 0.7f);  // IoU with A = 0, with B = 1/3
  const auto d = run(o, kIdentity, 640, 640, 0.25f, 0.30f);
  ASSERT_EQ(d.size(), 2u);
  EXPECT_FLOAT_EQ(d[0].score, 0.9f);
  EXPECT_FLOAT_EQ(d[1].score, 0.7f);
}

// ---------------------------------------------------------------- engine file

std::vector<char> with_ultralytics_header(const std::string& json, size_t payload_size) {
  const int32_t len = static_cast<int32_t>(json.size());
  std::vector<char> blob(4 + json.size() + payload_size, 'E');
  std::memcpy(blob.data(), &len, 4);
  std::memcpy(blob.data() + 4, json.data(), json.size());
  return blob;
}

TEST(EnginePayloadOffset, SkipsUltralyticsMetadataHeader) {
  const std::string meta = R"({"imgsz": [1280, 1280]})";
  EXPECT_EQ(engine_payload_offset(with_ultralytics_header(meta, 64)), 4 + meta.size());
}

TEST(EnginePayloadOffset, PlainEngineStartsAtZero) {
  EXPECT_EQ(engine_payload_offset(std::vector<char>(64, 'E')), 0u);
}

TEST(EnginePayloadOffset, TooShortBlobStartsAtZero) {
  EXPECT_EQ(engine_payload_offset(std::vector<char>(7, '{')), 0u);
}

TEST(EnginePayloadOffset, LengthBeyondFileIsIgnored) {
  auto blob = with_ultralytics_header("{}", 4);
  const int32_t huge = 1 << 20;
  std::memcpy(blob.data(), &huge, 4);
  EXPECT_EQ(engine_payload_offset(blob), 0u);
}

}  // namespace
