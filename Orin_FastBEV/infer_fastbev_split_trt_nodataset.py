import argparse
import os
import time
import numpy as np
import cv2

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def np_dtype_from_trt(dt: trt.DataType):
    if dt == trt.DataType.FLOAT: return np.float32
    if dt == trt.DataType.HALF:  return np.float16
    if dt == trt.DataType.INT8:  return np.int8
    if dt == trt.DataType.INT32: return np.int32
    if dt == trt.DataType.BOOL:  return np.bool_
    if hasattr(trt.DataType, "INT64") and dt == trt.DataType.INT64:
        return np.int64
    raise TypeError(f"Unsupported TRT dtype: {dt}")


class TRT10StaticRunner:
    def __init__(self, engine_path: str, log_level=trt.Logger.INFO):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        logger = trt.Logger(log_level)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.io_names if self.engine.get_tensor_mode(n) != trt.TensorIOMode.INPUT]

        print(f"\n[TRT] Loaded: {engine_path}")
        for i, n in enumerate(self.io_names):
            print(f"  [{i}] {n:15s} mode={self.engine.get_tensor_mode(n)} "
                  f"dtype={self.engine.get_tensor_dtype(n)} shape={self.engine.get_tensor_shape(n)}")

    def infer(self, feed: dict) -> dict:
        stream = cuda.Stream()

        # inputs
        d_in = {}
        for name, arr in feed.items():
            arr = np.ascontiguousarray(arr)
            d = cuda.mem_alloc(arr.nbytes)
            d_in[name] = d
            self.context.set_tensor_address(name, int(d))
            cuda.memcpy_htod_async(d, arr, stream)

        # outputs
        out_host = {}
        out_dev = {}
        for name in self.outputs:
            shape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
            dt = np_dtype_from_trt(self.engine.get_tensor_dtype(name))
            out_host[name] = np.empty(shape, dtype=dt)
            out_dev[name] = cuda.mem_alloc(out_host[name].nbytes)
            self.context.set_tensor_address(name, int(out_dev[name]))

        ok = self.context.execute_async_v3(stream_handle=stream.handle)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")

        for name in self.outputs:
            cuda.memcpy_dtoh_async(out_host[name], out_dev[name], stream)

        stream.synchronize()

        # free
        for d in d_in.values():
            d.free()
        for d in out_dev.values():
            d.free()

        return out_host


def preprocess_one(img_path: str, hw=(512, 1408)) -> np.ndarray:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"cv2.imread failed: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (hw[1], hw[0]))  # (W,H)

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    img = img.astype(np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)  # CHW


def build_input_f4_from_one(img_path: str, nv=6, frames=4, hw=(512, 1408)) -> np.ndarray:
    img_chw = preprocess_one(img_path, hw=hw)         # [3,H,W]
    one_frame = np.stack([img_chw] * nv, axis=0)      # [6,3,H,W]
    imgs24 = np.concatenate([one_frame] * frames, axis=0)  # [24,3,H,W]
    inp = imgs24.reshape(1, frames * nv, 3, hw[0], hw[1]).astype(np.float32, copy=False)
    return inp


def fake_build_mlvl_volume_from_pre(mlvl_feat: np.ndarray,
                                    frames=4, nv=6,
                                    X=250, Y=250, Z=6) -> np.ndarray:
    """
    mlvl_feat: (24,64,128,352) -> mlvl_volume: (1,256,250,250,6)
    """
    assert mlvl_feat.shape == (frames * nv, 64, 128, 352), f"unexpected mlvl_feat shape: {mlvl_feat.shape}"

    feat = mlvl_feat.reshape(frames, nv, 64, 128, 352)
    vol = np.empty((1, frames * 64, X, Y, Z), dtype=np.float32)

    for f in range(frames):
        cams = feat[f]                # (6,64,128,352)
        fused = cams.mean(axis=0)     # (64,128,352)

        # resize (128,352) -> (250,250) on CPU
        fused_hwc = np.ascontiguousarray(fused.transpose(1, 2, 0))  # (128,352,64)
        resized_hwc = cv2.resize(fused_hwc, (Y, X), interpolation=cv2.INTER_LINEAR)  # (250,250,64)
        resized = resized_hwc.transpose(2, 0, 1).astype(np.float32, copy=False)      # (64,250,250)

        vol[0, f*64:(f+1)*64, :, :, :] = resized[:, :, :, None]  # broadcast Z=6

    return vol


def apply_dir_correction(bboxes: np.ndarray, dir_cls: np.ndarray) -> np.ndarray:
    """
    可选：把 dir_cls(0/1) 用于 yaw 修正。
    注意：不同实现 dir 的定义可能不同，这里给一个常见修正方式：
    若 dir_cls==1，则 yaw += pi
    """
    b = bboxes.copy()
    yaw = b[:, 6]
    yaw = yaw + (dir_cls.astype(np.float32) * np.pi)
    # normalize to [-pi, pi]
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    b[:, 6] = yaw
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_engine", default="fastbev_pre.engine")
    ap.add_argument("--post_engine", default="fastbev_post.engine")
    ap.add_argument("--img", default="0-FRONT.jpg", help="single image path, will be replicated to 24")
    ap.add_argument("--nv", type=int, default=6)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--score_thr", type=float, default=0.2)
    ap.add_argument("--dir_correction", action="store_true")
    args = ap.parse_args()

    for p in [args.pre_engine, args.post_engine, args.img]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    pre = TRT10StaticRunner(args.pre_engine)
    post = TRT10StaticRunner(args.post_engine)

    # ---- pre input ----
    inp = build_input_f4_from_one(args.img, nv=args.nv, frames=args.frames)

    exp_pre_in = tuple(int(x) for x in pre.engine.get_tensor_shape("image"))
    if tuple(inp.shape) != exp_pre_in:
        raise RuntimeError(f"pre input shape mismatch: got {inp.shape}, expected {exp_pre_in}")

    # ---- run pre ----
    t0 = time.time()
    pre_in_dtype = np_dtype_from_trt(pre.engine.get_tensor_dtype("image"))
    pre_out = pre.infer({"image": inp.astype(pre_in_dtype, copy=False)})
    mlvl_feat = pre_out["mlvl_feat"]  # (24,64,128,352)
    t1 = time.time()

    print(f"\n[Pre] mlvl_feat: {mlvl_feat.shape} {mlvl_feat.dtype} mean {float(mlvl_feat.mean()):.6f} max {float(mlvl_feat.max()):.6f}")

    # ---- fake middle ----
    mlvl_volume = fake_build_mlvl_volume_from_pre(mlvl_feat, frames=args.frames, nv=args.nv)
    exp_post_in = tuple(int(x) for x in post.engine.get_tensor_shape("mlvl_volume"))
    if tuple(mlvl_volume.shape) != exp_post_in:
        raise RuntimeError(f"post input shape mismatch: got {mlvl_volume.shape}, expected {exp_post_in}")
    t2 = time.time()

    # ---- run post ----
    post_in_dtype = np_dtype_from_trt(post.engine.get_tensor_dtype("mlvl_volume"))
    post_out = post.infer({"mlvl_volume": mlvl_volume.astype(post_in_dtype, copy=False)})
    t3 = time.time()

    scores = post_out["scores"]     # (1000,10)
    bboxes = post_out["bboxes"]     # (1000,9)
    dir_cls = post_out["dir_cls"]   # (1000,) int64

    # optional yaw correction
    if args.dir_correction:
        bboxes = apply_dir_correction(bboxes, dir_cls)

    # filter by score_thr (use max class score)
    max_scores = scores.max(axis=1)
    keep = np.where(max_scores >= args.score_thr)[0]

    print("\n[Post] outputs:")
    print("  scores:", scores.shape, scores.dtype)
    print("  bboxes:", bboxes.shape, bboxes.dtype)
    print("  dir_cls:", dir_cls.shape, dir_cls.dtype)

    print(f"\nKeep {len(keep)}/{len(max_scores)} by score_thr={args.score_thr}")

    # show top10 kept
    order = keep[np.argsort(-max_scores[keep])][:10]
    print("\nTop-10 (仅用于跑通链路，结果不具备意义):")
    for i, idx in enumerate(order):
        cls_id = int(np.argmax(scores[idx]))
        print(f"#{i:02d} idx={idx:4d} score={max_scores[idx]:.4f} cls={cls_id} dir={int(dir_cls[idx])} box={bboxes[idx]}")

    print(f"\nTiming: pre={t1-t0:.3f}s, fake_mid={t2-t1:.3f}s, post={t3-t2:.3f}s, total={t3-t0:.3f}s")


if __name__ == "__main__":
    main()

