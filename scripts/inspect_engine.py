"""Print the Ultralytics metadata and TensorRT input/output shapes of an .engine file."""
import json
import sys

import tensorrt as trt

for path in sys.argv[1:]:
    blob = open(path, "rb").read()
    n = int.from_bytes(blob[:4], "little", signed=True)
    meta = json.loads(blob[4:4 + n]) if 0 < n < len(blob) and blob[4:5] == b"{" else {}
    payload = blob[4 + n:] if meta else blob
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(payload)
    shapes = {engine.get_tensor_name(i): tuple(engine.get_tensor_shape(engine.get_tensor_name(i)))
              for i in range(engine.num_io_tensors)}
    print(path)
    print("  metadata: imgsz", meta.get("imgsz"), "| dynamic", meta.get("args", {}).get("dynamic"),
          "| half", meta.get("args", {}).get("half"), "| int8", meta.get("args", {}).get("int8"))
    print("  tensors:", shapes)
