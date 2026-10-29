#pragma once

// Letterbox geometry, matching Ultralytics' LetterBox (centred padding, value 114).
// Plain C++ (no CUDA headers) so it can be unit-tested on a CPU-only machine.
struct Letterbox {
  float gain;   // resize ratio applied to the source image
  int new_w;    // resized (unpadded) width
  int new_h;    // resized (unpadded) height
  int left;     // horizontal padding before the image
  int top;      // vertical padding before the image
};

Letterbox compute_letterbox(int src_w, int src_h, int dst_w, int dst_h);
