#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import torch
import tensorrt as trt

def die(msg):
    print("\n[FAIL]", msg)
    sys.exit(1)

def ok(msg):
    print("[OK]", msg)

def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT8:
        return torch.int8
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.BOOL:
        return torch.bool
    if hasattr(trt.DataType, "INT64") and dtype == trt.DataType.INT64:
        return torch.int64
    raise TypeError(f"Unsupported TRT dtype: {dtype}")

class TRTRunnerTRT10:
    def __init__(self, engine_path: str, device: str = "cuda:0", log_level=trt.Logger.ERROR):
        self.device = torch.device(device)

        logger = trt.Logger(log_level)
        runtime = trt.Runtime(logger)

        if not os.path.exists(engine_path):
            die(f"Engine not found: {engine_path}")

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            die(f"Failed to deserialize engine: {engine_path}")

        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            die("Failed to create execution context")

        self.io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.input_names = [n for n in self.io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.io_names if engine.get_tensor_mode(n) != trt.TensorIOMode.INPUT]

        print(f"\n[TRT] Loaded: {engine_path}")
        for i, n in enumerate(self.io_names):
            mode = engine.get_tensor_mode(n)
            dt = engine.get_tensor_dtype(n)
            shp = tuple(int(x) for x in engine.get_tensor_shape(n))
            print(f"  [{i}] {n:18s} mode={mode} dtype={dt} shape={shp}")

    def infer(self, inputs):
        # set inputs
        for name, t in inputs.items():
            if name not in self.input_names:
                die(f"'{name}' not in engine inputs: {self.input_names}")
            if t.device != self.device:
                t = t.to(self.device)
            t = t.contiguous()

            exp_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            if t.dtype != exp_dtype:
                t = t.to(exp_dtype)

            okk = self.context.set_input_shape(name, tuple(int(x) for x in t.shape))
            if not okk:
                die(f"set_input_shape failed for {name}: {tuple(t.shape)}")

            self.context.set_tensor_address(name, int(t.data_ptr()))
            inputs[name] = t

        # allocate outputs
        outputs = {}
        for name in self.output_names:
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                die(f"Output {name} dynamic shape {shape} - check profile/input shapes")
            dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            out = torch.empty(shape, device=self.device, dtype=dtype)
            self.context.set_tensor_address(name, int(out.data_ptr()))
            outputs[name] = out

        stream = torch.cuda.current_stream(self.device).cuda_stream
        okk = self.context.execute_async_v3(stream_handle=stream)
        if not okk:
            die("execute_async_v3 returned False")
        return outputs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_engine", default="fastbev_pre.engine")
    ap.add_argument("--post_engine", default="fastbev_post.engine")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img_h", type=int, default=512)
    ap.add_argument("--img_w", type=int, default=1408)
    ap.add_argument("--nv", type=int, default=6)
    ap.add_argument("--frames", type=int, default=4, help="f4 => 4 frames, so N=24")
    ap.add_argument("--bev_x", type=int, default=250)
    ap.add_argument("--bev_y", type=int, default=250)
    ap.add_argument("--bev_z", type=int, default=6)
    ap.add_argument("--cvol", type=int, default=256, help="post input channels, f4 typically 256")
    args = ap.parse_args()

    print("=== Basic env ===")
    print("python:", sys.version.replace("\n", " "))
    print("numpy :", np.__version__)
    if int(np.__version__.split(".")[0]) >= 2:
        die("NumPy >=2 detected. Your Jetson torch build typically requires numpy<2 (e.g. 1.26.4).")

    print("torch :", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        die("torch.cuda not available")

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    ok(f"using device {dev}")

    # quick CUDA compute test
    x = torch.randn(1024, 1024, device=dev)
    y = x @ x.t()
    torch.cuda.synchronize()
    ok(f"matmul OK, y.mean={float(y.mean()):.6f}")

    print("\n=== Load engines ===")
    pre = TRTRunnerTRT10(args.pre_engine, device=args.device, log_level=trt.Logger.INFO)
    post = TRTRunnerTRT10(args.post_engine, device=args.device, log_level=trt.Logger.INFO)

    # fake input for pre: (1, N, 3, H, W)
    N = args.nv * args.frames
    img = torch.randn((1, N, 3, args.img_h, args.img_w), device=dev, dtype=torch.float32)

    pre_in = pre.input_names[0]
    t0 = time.time()
    pre_out = pre.infer({pre_in: img})
    torch.cuda.synchronize()
    t1 = time.time()

    pre_out_name = pre.output_names[0]
    feat = pre_out[pre_out_name]
    ok(f"pre out {pre_out_name} shape={tuple(feat.shape)} dtype={feat.dtype} time={t1-t0:.3f}s")
    print(f"  mean={float(feat.float().mean()):.6f} max={float(feat.float().max()):.6f} min={float(feat.float().min()):.6f}")

    # fake mid: dummy BEV volume to feed post
    vol = torch.randn((1, args.cvol, args.bev_x, args.bev_y, args.bev_z), device=dev, dtype=torch.float32)

    post_in = post.input_names[0]
    t2 = time.time()
    post_out = post.infer({post_in: vol})
    torch.cuda.synchronize()
    t3 = time.time()

    ok(f"post time={t3-t2:.3f}s outputs:")
    for k, v in post_out.items():
        print(f"  - {k:12s} shape={tuple(v.shape)} dtype={v.dtype}")
        if v.numel() > 0 and v.dtype in (torch.float16, torch.float32):
            print(f"    mean={float(v.float().mean()):.6f} max={float(v.float().max()):.6f}")

    print("\n=== Summary ===")
    ok("TensorRT engines load OK")
    ok("pre inference OK (fake images)")
    ok("post inference OK (fake BEV volume)")
    print("\nNOTE: This check validates environment + engines runtime only.")
    print("      Real accuracy requires real nuscenes infos.pkl + real backproject.")

if __name__ == "__main__":
    main()
