#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import pickle
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import tensorrt as trt


# -----------------------------
# TensorRT10 runner (torch buffers)
# -----------------------------
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
    """
    TRT10+ API: set_input_shape / set_tensor_address / execute_async_v3
    Uses torch.cuda tensors as buffers (no pycuda).
    """
    def __init__(self, engine_path: str, device: str = "cuda:0", log_level=trt.Logger.ERROR):
        self.device = torch.device(device)

        logger = trt.Logger(log_level)
        runtime = trt.Runtime(logger)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.input_names = [n for n in self.io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.io_names if engine.get_tensor_mode(n) != trt.TensorIOMode.INPUT]

        print(f"\n[TRT] Loaded: {engine_path}")
        for i, n in enumerate(self.io_names):
            mode = engine.get_tensor_mode(n)
            dt = engine.get_tensor_dtype(n)
            shp = tuple(int(x) for x in engine.get_tensor_shape(n))
            print(f"  [{i}] {n:18s} mode={mode} dtype={dt} shape={shp}")

    def infer(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # set inputs
        for name, t in inputs.items():
            if name not in self.input_names:
                raise KeyError(f"'{name}' not in engine inputs. Inputs: {self.input_names}")
            if t.device != self.device:
                t = t.to(self.device)
            t = t.contiguous()

            exp_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            if t.dtype != exp_dtype:
                t = t.to(exp_dtype)

            ok = self.context.set_input_shape(name, tuple(int(x) for x in t.shape))
            if not ok:
                raise RuntimeError(f"set_input_shape failed for {name} shape={tuple(t.shape)}")

            self.context.set_tensor_address(name, int(t.data_ptr()))
            inputs[name] = t

        # allocate outputs
        outputs: Dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Output {name} has dynamic shape {shape}. Check input shapes/profile.")
            dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            out = torch.empty(shape, device=self.device, dtype=dtype)
            self.context.set_tensor_address(name, int(out.data_ptr()))
            outputs[name] = out

        stream = torch.cuda.current_stream(self.device).cuda_stream
        ok = self.context.execute_async_v3(stream_handle=stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        return outputs


# -----------------------------
# Image preprocess
# -----------------------------
def preprocess_image(
    img_path: str,
    hw: Tuple[int, int] = (512, 1408),
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
    rgb: bool = True,
) -> np.ndarray:
    """Return float32 CHW normalized."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32)
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    img = img.transpose(2, 0, 1)  # CHW
    return img


# -----------------------------
# Backproject (official-like math)
# -----------------------------
@torch.no_grad()
def get_points(
    n_voxels_xyz: Tuple[int, int, int],
    voxel_size_xyz: Tuple[float, float, float],
    origin: torch.Tensor
) -> torch.Tensor:
    nx, ny, nz = n_voxels_xyz
    voxel_size = torch.tensor(voxel_size_xyz, dtype=torch.float32, device=origin.device)
    n_voxels = torch.tensor([nx, ny, nz], dtype=torch.float32, device=origin.device)

    try:
        grids = torch.meshgrid(
            torch.arange(nx, device=origin.device),
            torch.arange(ny, device=origin.device),
            torch.arange(nz, device=origin.device),
            indexing="ij",
        )
    except TypeError:
        grids = torch.meshgrid(
            torch.arange(nx, device=origin.device),
            torch.arange(ny, device=origin.device),
            torch.arange(nz, device=origin.device),
        )

    points = torch.stack(grids)  # [3,nx,ny,nz]
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def compute_projection(
    intrinsic: np.ndarray,
    extrinsics: List[np.ndarray],
    stride: int,
    device: torch.device
) -> torch.Tensor:
    """
    intrinsic: (3,3) or (4,4)
    extrinsics: list of (4,4) or (3,4) length nv
    returns: [nv,3,4]
    """
    intrinsic = intrinsic.astype(np.float32)
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    K = torch.tensor(intrinsic, dtype=torch.float32, device=device)
    K[:2] /= float(stride)

    projs = []
    for ext in extrinsics:
        ext = ext.astype(np.float32)
        if ext.shape == (4, 4):
            ext = ext[:3, :]
        elif ext.shape != (3, 4):
            raise ValueError(f"Unexpected extrinsic shape: {ext.shape}")
        E = torch.tensor(ext, dtype=torch.float32, device=device)
        projs.append(K @ E)  # [3,4]
    return torch.stack(projs, dim=0)  # [nv,3,4]


def backproject_inplace(features: torch.Tensor, points: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """
    features: [nv,C,Hf,Wf]
    points: [3,nx,ny,nz]
    projection: [nv,3,4]
    -> volume: [C,nx,ny,nz]
    """
    nv, c, h, w = features.shape
    pts = points.view(1, 3, -1).expand(nv, 3, -1)
    pts = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)  # [nv,4,N]

    pts2d3 = torch.bmm(projection, pts)  # [nv,3,N]
    x = (pts2d3[:, 0] / pts2d3[:, 2]).round().long()
    y = (pts2d3[:, 1] / pts2d3[:, 2]).round().long()
    z = pts2d3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < w) & (y < h) & (z > 0)

    volume = torch.zeros((c, pts.shape[-1]), device=features.device, dtype=features.dtype)
    for i in range(nv):
        vi = valid[i]
        volume[:, vi] = features[i, :, y[i, vi], x[i, vi]]
    nx, ny, nz = points.shape[-3:]
    return volume.view(c, nx, ny, nz)


def build_mlvl_volume(
    image_input: torch.Tensor,        # [1,N,3,H,W]
    lidar2img_meta: Dict[str, Any],   # dict with intrinsic/extrinsic/origin
    mlvl_feat_nchw: torch.Tensor,     # [N,C,Hf,Wf]
    nv: int,
    n_voxels_xyz: Tuple[int, int, int],
    voxel_size_xyz: Tuple[float, float, float],
    use_frames_concat: bool = True,
) -> torch.Tensor:
    """
    Returns [1, Cvol, X, Y, Z]
    Assumes N = frames*nv and lidar2img_meta contains:
      intrinsic: (3,3)/(4,4)
      extrinsic: list length N (each 4x4 or 3x4)
      origin: (3,)
    """
    device = mlvl_feat_nchw.device
    N, C, Hf, Wf = mlvl_feat_nchw.shape
    if N % nv != 0:
        raise ValueError(f"N={N} not divisible by nv={nv}")
    frames = N // nv

    stride = math.ceil(int(image_input.shape[-1]) / int(Wf))

    intrinsic = np.array(lidar2img_meta["intrinsic"], dtype=np.float32)
    extr_all = lidar2img_meta["extrinsic"]
    origin_np = np.array(lidar2img_meta["origin"], dtype=np.float32).reshape(-1)
    if origin_np.shape[0] != 3:
        raise ValueError(f"origin must be (3,), got {origin_np.shape}")
    origin = torch.tensor(origin_np, dtype=torch.float32, device=device)

    points = get_points(n_voxels_xyz, voxel_size_xyz, origin).to(device)

    feat_5d = mlvl_feat_nchw.view(frames, nv, C, Hf, Wf)  # [T,nv,C,Hf,Wf]
    vols = []
    frame_ids = list(range(frames)) if use_frames_concat else [frames - 1]

    for t in frame_ids:
        extr = [np.array(x, dtype=np.float32) for x in extr_all[t * nv:(t + 1) * nv]]
        proj = compute_projection(intrinsic, extr, stride=stride, device=device)
        vol = backproject_inplace(feat_5d[t], points, proj)  # [C,X,Y,Z]
        vols.append(vol)

    vol_cat = torch.cat(vols, dim=0)  # [Cvol,X,Y,Z]
    return vol_cat.unsqueeze(0)       # [1,Cvol,X,Y,Z]


# -----------------------------
# infos.pkl reading & flexible field extraction
# -----------------------------
def load_infos(pkl_path: str) -> List[dict]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "infos" in obj:
        return obj["infos"]
    if isinstance(obj, list):
        return obj
    raise TypeError(f"Unknown infos format: {type(obj)}")


def make_abs(data_root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(data_root, path)


def try_get(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def extract_from_info(
    info: dict,
    data_root: str,
    nv: int,
    expected_N: int,
    cam_order: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Return:
      img_paths: list length N
      lidar2img_meta: {intrinsic, extrinsic(list length N), origin}
    Supports several common formats produced by create_data.py pipelines.
    """

    # -------- Case 1: top-level lidar2img + img_paths (already expanded for f4) --------
    if "lidar2img" in info and isinstance(info["lidar2img"], dict):
        l2i = info["lidar2img"]
        intr = try_get(l2i, ["intrinsic", "K"])
        extr = try_get(l2i, ["extrinsic", "E", "extrinsics"])
        origin = try_get(l2i, ["origin"])
        img_paths_rel = try_get(l2i, ["img_paths", "image_paths"]) or try_get(info, ["img_paths", "image_paths"])

        if intr is not None and extr is not None and origin is not None and img_paths_rel is not None:
            img_paths = [make_abs(data_root, p) for p in img_paths_rel]
            extr_list = [np.array(e, dtype=np.float32) for e in extr]
            meta = {
                "intrinsic": np.array(intr, dtype=np.float32),
                "extrinsic": extr_list,
                "origin": np.array(origin, dtype=np.float32),
            }
            return img_paths, meta

    # -------- Case 2: info['cams'] dict (usually 6 cams single frame) --------
    if "cams" in info and isinstance(info["cams"], dict):
        cams = info["cams"]

        # try build from 6 cams
        if all(k in cams for k in cam_order):
            img_paths = []
            extr_list = []
            intr = None
            origin = None

            for cam in cam_order:
                cd = cams[cam]
                dp = try_get(cd, ["data_path", "img_path", "filename"])
                if dp is None:
                    raise KeyError(f"cam {cam} has no data_path/img_path/filename")
                img_paths.append(make_abs(data_root, dp))

                # lidar2img nested
                if "lidar2img" in cd and isinstance(cd["lidar2img"], dict):
                    if intr is None:
                        intr = np.array(cd["lidar2img"]["intrinsic"], dtype=np.float32)
                    ext = np.array(cd["lidar2img"]["extrinsic"], dtype=np.float32)
                    extr_list.append(ext)
                    if origin is None and "origin" in cd["lidar2img"]:
                        origin = np.array(cd["lidar2img"]["origin"], dtype=np.float32)
                else:
                    # separate intrinsic/extrinsic
                    if intr is None:
                        intr = np.array(try_get(cd, ["intrinsic", "K"]), dtype=np.float32)
                    ext = try_get(cd, ["extrinsic", "E"])
                    if ext is None:
                        raise KeyError(f"cam {cam} has no extrinsic/E")
                    extr_list.append(np.array(ext, dtype=np.float32))
                    if origin is None:
                        origin = try_get(cd, ["origin"])
                        if origin is not None:
                            origin = np.array(origin, dtype=np.float32)

            if intr is None or origin is None:
                raise KeyError("Cannot get intrinsic/origin from cams format.")
            meta = {"intrinsic": intr, "extrinsic": extr_list, "origin": origin}

            # if engine expects 24 but only 6 provided, we cannot invent sweeps here.
            # user should use f1 engine or infos that includes f4 expanded paths.
            if expected_N != len(img_paths):
                raise RuntimeError(
                    f"infos provides N={len(img_paths)} images (single frame), "
                    f"but engine expects N={expected_N}. "
                    "=> 你现在的 pre.engine 是 f4(N=24)，需要 infos 里也提供 24 张图（含 sweeps）。"
                )

            return img_paths, meta

    # -------- Fallback: dump keys for debugging --------
    raise KeyError(
        "Unsupported infos format. Debug:\n"
        f"info keys: {list(info.keys())}\n"
        f"cams keys (if any): {list(info.get('cams', {}).keys())[:20] if isinstance(info.get('cams', None), dict) else None}\n"
        "Try printing one info dict and I adapt the parser."
    )


# -----------------------------
# Simple dir correction (optional)
# -----------------------------
def limit_period(val: torch.Tensor, offset: float = 0.5, period: float = math.pi) -> torch.Tensor:
    return val - torch.floor(val / period + offset) * period


def apply_dir_correction(bboxes: torch.Tensor, dir_cls: torch.Tensor, dir_offset: float, dir_limit_offset: float) -> torch.Tensor:
    yaw = bboxes[:, 6]
    yaw = limit_period(yaw - dir_offset, dir_limit_offset, math.pi)
    yaw = yaw + dir_offset + math.pi * dir_cls.to(yaw.dtype)
    bboxes[:, 6] = yaw
    return bboxes


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="nuscenes_infos_*.pkl (val/train)")
    ap.add_argument("--data_root", required=True, help="nuScenes root (contains samples/, sweeps/, v1.0-mini/)")
    ap.add_argument("--pre_engine", required=True)
    ap.add_argument("--post_engine", required=True)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--nv", type=int, default=6)
    ap.add_argument("--img_h", type=int, default=512)
    ap.add_argument("--img_w", type=int, default=1408)
    ap.add_argument("--n_voxels", type=int, nargs=3, default=[250, 250, 6])      # X Y Z
    ap.add_argument("--voxel_size", type=float, nargs=3, default=[0.5, 0.5, 1.5])
    ap.add_argument("--score_thr", type=float, default=0.2)
    ap.add_argument("--dir_correction", action="store_true")
    ap.add_argument("--dir_offset", type=float, default=0.0)
    ap.add_argument("--dir_limit_offset", type=float, default=0.0)
    args = ap.parse_args()

    # sanity: numpy<2
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError(f"NumPy is {np.__version__}, please use numpy==1.26.4")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # load infos
    infos = load_infos(args.infos)
    if args.sample_idx >= len(infos):
        raise IndexError(f"sample_idx {args.sample_idx} out of range. len={len(infos)}")
    info = infos[args.sample_idx]

    # load engines first to know expected N/Cvol
    pre = TRTRunnerTRT10(args.pre_engine, device=args.device, log_level=trt.Logger.INFO)
    post = TRTRunnerTRT10(args.post_engine, device=args.device, log_level=trt.Logger.INFO)

    pre_in = pre.input_names[0]
    pre_in_shape = tuple(int(x) for x in pre.engine.get_tensor_shape(pre_in))
    if any(d <= 0 for d in pre_in_shape):
        raise RuntimeError(f"pre engine input shape is dynamic: {pre_in_shape} (this script assumes fixed).")
    expected_N = int(pre_in_shape[1])

    post_in = post.input_names[0]
    post_in_shape = tuple(int(x) for x in post.engine.get_tensor_shape(post_in))
    if any(d <= 0 for d in post_in_shape):
        raise RuntimeError(f"post engine input shape is dynamic: {post_in_shape} (this script assumes fixed).")
    expected_Cvol = int(post_in_shape[1])
    expected_X, expected_Y, expected_Z = int(post_in_shape[2]), int(post_in_shape[3]), int(post_in_shape[4])

    if tuple(args.n_voxels) != (expected_X, expected_Y, expected_Z):
        print(f"[Warn] args.n_voxels={tuple(args.n_voxels)} != post expects {(expected_X, expected_Y, expected_Z)}. Using engine shape.")
        args.n_voxels = [expected_X, expected_Y, expected_Z]

    # cam order should match training pipeline
    cam_order = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]

    # extract N images + meta
    img_paths, l2i_meta = extract_from_info(
        info=info,
        data_root=args.data_root,
        nv=args.nv,
        expected_N=expected_N,
        cam_order=cam_order,
    )

    if len(img_paths) != expected_N:
        raise RuntimeError(f"Collected {len(img_paths)} images, but engine expects N={expected_N}")

    if len(l2i_meta["extrinsic"]) != expected_N:
        raise RuntimeError(f"extrinsic list len={len(l2i_meta['extrinsic'])} but expected N={expected_N}")

    print(f"\n[Infos] sample_idx={args.sample_idx} N(images)={len(img_paths)}")
    print(f"  first img: {img_paths[0]}")

    # build image input [1,N,3,H,W]
    imgs = np.zeros((1, expected_N, 3, args.img_h, args.img_w), dtype=np.float32)
    for i, p in enumerate(img_paths):
        imgs[0, i] = preprocess_image(p, hw=(args.img_h, args.img_w))
    image_input = torch.from_numpy(imgs).to(device)

    # run pre
    pre_out = pre.infer({pre_in: image_input})
    pre_out_name = pre.output_names[0]
    mlvl_feat = pre_out[pre_out_name]  # expected [N,C,Hf,Wf]
    torch.cuda.synchronize()
    print(f"\n[Pre] {pre_out_name}: shape={tuple(mlvl_feat.shape)} dtype={mlvl_feat.dtype} mean={float(mlvl_feat.float().mean()):.6f}")

    if mlvl_feat.dim() != 4:
        raise RuntimeError(f"Unexpected pre output dim={mlvl_feat.dim()}, shape={tuple(mlvl_feat.shape)}")
    if mlvl_feat.shape[0] != expected_N:
        raise RuntimeError(f"pre output first dim={mlvl_feat.shape[0]} != expected_N={expected_N}")

    feat_nchw = mlvl_feat  # [N,C,Hf,Wf]
    Cfeat = int(feat_nchw.shape[1])
    frames = expected_N // args.nv
    if expected_N % args.nv != 0:
        raise RuntimeError(f"expected_N={expected_N} not divisible by nv={args.nv}")

    # determine concat mode by expected_Cvol
    if expected_Cvol == frames * Cfeat:
        use_frames_concat = True
    elif expected_Cvol == Cfeat:
        use_frames_concat = False
        print("[Mid] Post expects Cvol=Cfeat -> will use ONLY last frame for volume.")
    else:
        raise RuntimeError(f"Cvol mismatch: post expects {expected_Cvol}, but frames*Cfeat={frames*Cfeat}, Cfeat={Cfeat}")

    # build real mlvl_volume
    mlvl_volume = build_mlvl_volume(
        image_input=image_input,
        lidar2img_meta=l2i_meta,
        mlvl_feat_nchw=feat_nchw,
        nv=args.nv,
        n_voxels_xyz=tuple(args.n_voxels),
        voxel_size_xyz=tuple(args.voxel_size),
        use_frames_concat=use_frames_concat,
    )
    torch.cuda.synchronize()
    print(f"[Mid] mlvl_volume: shape={tuple(mlvl_volume.shape)} dtype={mlvl_volume.dtype}")

    # run post
    post_out = post.infer({post_in: mlvl_volume})
    torch.cuda.synchronize()
    print("\n[Post] outputs:")
    for k, v in post_out.items():
        print(f"  {k:12s} shape={tuple(v.shape)} dtype={v.dtype}")

    # pick outputs by name
    def pick(cands, default_idx):
        for n in cands:
            if n in post_out:
                return post_out[n]
        return post_out[post.output_names[default_idx]]

    scores = pick(["scores", "cls_score"], 0).float()     # [K,num_cls]
    bboxes = pick(["bboxes", "boxes"], 1).float()         # [K,9]
    dir_cls = pick(["dir_cls", "dir_scores"], 2)          # [K]

    max_scores, labels = scores.max(dim=1)
    keep = max_scores > float(args.score_thr)

    if keep.sum().item() == 0:
        print(f"\n[Result] No boxes above score_thr={args.score_thr}")
        return

    max_scores = max_scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]
    dir_cls = dir_cls[keep].long()

    # optional yaw correction
    if args.dir_correction:
        bboxes = apply_dir_correction(bboxes, dir_cls, dir_offset=float(args.dir_offset), dir_limit_offset=float(args.dir_limit_offset))

    order = torch.argsort(max_scores, descending=True)
    max_scores = max_scores[order]
    labels = labels[order]
    bboxes = bboxes[order]
    dir_cls = dir_cls[order]

    topn = min(20, max_scores.numel())
    print(f"\n[Result] top{topn} (score_thr={args.score_thr})")
    for i in range(topn):
        print(f"  #{i:02d} score={float(max_scores[i]):.3f} cls={int(labels[i])} dir={int(dir_cls[i])} "
              f"box={np.array2string(bboxes[i].cpu().numpy(), precision=3, floatmode='fixed')}")

    print("\nDone. (This is TopK+decode; for full metrics you still need proper rotated NMS + official eval.)")


if __name__ == "__main__":
    main()
