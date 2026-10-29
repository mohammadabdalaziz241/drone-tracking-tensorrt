// compute_letterbox must reproduce Ultralytics' LetterBox geometry exactly, otherwise the C++
// and Python paths feed the engine different pixels.
//
// Expected values were generated from the formula in ultralytics==8.4.184
// (ultralytics/data/augment.py, LetterBox with auto=False, scaleup=True, center=True; float64
// arithmetic and Python's round), not from this C++ code.

#include <gtest/gtest.h>

#include <string>

#include "letterbox.h"

namespace {

struct Case {
  int src_w, src_h, dst;  // square network input, as in this project (640 or 1280)
  int new_w, new_h, left, top;
};

class LetterboxMatchesUltralytics : public ::testing::TestWithParam<Case> {};

TEST_P(LetterboxMatchesUltralytics, Geometry) {
  const Case c = GetParam();
  const Letterbox lb = compute_letterbox(c.src_w, c.src_h, c.dst, c.dst);
  EXPECT_EQ(lb.new_w, c.new_w);
  EXPECT_EQ(lb.new_h, c.new_h);
  EXPECT_EQ(lb.left, c.left);
  EXPECT_EQ(lb.top, c.top);
  // The resized image plus padding never exceeds the network input.
  EXPECT_LE(lb.left + lb.new_w, c.dst);
  EXPECT_LE(lb.top + lb.new_h, c.dst);
}

INSTANTIATE_TEST_SUITE_P(
    VisDroneAndEdgeSizes, LetterboxMatchesUltralytics,
    ::testing::Values(
        // Typical VisDrone resolutions.
        Case{1360, 765, 640, 640, 360, 0, 140}, Case{1360, 765, 1280, 1280, 720, 0, 280},
        Case{1920, 1080, 640, 640, 360, 0, 140}, Case{1920, 1080, 1280, 1280, 720, 0, 280},
        Case{960, 540, 640, 640, 360, 0, 140}, Case{960, 540, 1280, 1280, 720, 0, 280},
        Case{1400, 1050, 640, 640, 480, 0, 80}, Case{1400, 1050, 1280, 1280, 960, 0, 160},
        Case{2000, 1500, 640, 640, 480, 0, 80}, Case{2000, 1500, 1280, 1280, 960, 0, 160},
        Case{1344, 756, 640, 640, 360, 0, 140}, Case{1344, 756, 1280, 1280, 720, 0, 280},
        Case{480, 360, 640, 640, 480, 0, 80}, Case{480, 360, 1280, 1280, 960, 0, 160},
        Case{1400, 788, 640, 640, 360, 0, 140}, Case{1400, 788, 1280, 1280, 720, 0, 280},
        Case{1916, 1078, 640, 640, 360, 0, 140}, Case{1916, 1078, 1280, 1280, 720, 0, 280},
        // Odd padding: the extra pixel goes to the bottom/right, as in Ultralytics.
        Case{100, 99, 640, 640, 634, 0, 3}, Case{100, 99, 1280, 1280, 1267, 0, 6},
        // Very thin image, padded mostly on the left/right.
        Case{7, 640, 640, 7, 640, 316, 0}, Case{7, 640, 1280, 14, 1280, 633, 0}),
    [](const ::testing::TestParamInfo<Case>& info) {
      const Case& c = info.param;
      return std::to_string(c.src_w) + "x" + std::to_string(c.src_h) + "_to_" + std::to_string(c.dst);
    });

TEST(Letterbox, GainIsTheLimitingRatio) {
  const Letterbox lb = compute_letterbox(1920, 1080, 1280, 1280);
  EXPECT_FLOAT_EQ(lb.gain, 1280.f / 1920.f);
}

TEST(Letterbox, UpscalesSmallImages) {
  const Letterbox lb = compute_letterbox(320, 320, 640, 640);
  EXPECT_FLOAT_EQ(lb.gain, 2.f);
  EXPECT_EQ(lb.new_w, 640);
  EXPECT_EQ(lb.new_h, 640);
  EXPECT_EQ(lb.left, 0);
  EXPECT_EQ(lb.top, 0);
}

}  // namespace
