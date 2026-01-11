#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import tensorrt as trt


# =========================================================
# Utils
# =========================================================
def ensure_numpy_1x():
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        raise RuntimeError(
            f"NumPy is {np.__version__} but your torch build requires NumPy 1.x.\n"
            f"Fix: pip install -U 'numpy==1.26.4'"
        )


def mkdir_p(p: str):
    os.makedirs(p, exist_ok=True)


def normalize_path(data_root: str, p: str) -> str:
    """Fix common infos.pkl path prefixes like './data/nuscenes/...'."""
    if p is None:
        return p
    p = str(p)
    if os.path.isabs(p) and os.path.exists(p):
        return p

    p2 = p.replace("\\", "/")
    # strip leading "./"
    if p2.startswith("./"):
        p2 = p2[2:]

    # common prefix inside pkl
    # e.g. "data/nuscenes/samples/....jpg" or "./data/nuscenes/...."
    key = "data/nuscenes/"
    if key in p2:
        p2 = p2.split(key, 1)[1]  # keep after "data/nuscenes/"

    full = os.path.join(data_root, p2)
    return full


def load_infos(pkl_path: str):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "infos" in obj:
        return obj["infos"]
    if isinstance(obj, list):
        return obj
    raise TypeError(f"Unknown infos format: {type(obj)}")


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


# =========================================================
# TRT10 Runner (torch buffers, no pycuda)
# =========================================================
class TRTRunnerTRT10:
    def __init__(self, engine_path: str, device: str = "cuda:0", log_level=trt.Logger.INFO):
        self.device = torch.device(device)
        logger = trt.Logger(log_level)
        runtime = trt.Runtime(logger)
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


# =========================================================
# Config-aligned image preprocessing (test path)
# Your config:
#   src_size=(900,1600)
#   input_size=(512,1408)
#   test_resize=0.0, test_rotate=0.0, test_flip=False
#   pad=(0,0,0,0)
#   Normalize: mean/std, to_rgb=True
# =========================================================
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

def preprocess_image_and_K_test(
    img_path: str,
    K: np.ndarray,
    src_size_hw: Tuple[int, int] = (900, 1600),
    input_size_hw: Tuple[int, int] = (512, 1408),
    test_resize: float = 0.0,
    test_flip: bool = False,
    test_rotate_deg: float = 0.0,
    pad: Tuple[int, int, int, int] = (0, 0, 0, 0),
    to_rgb: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      img_chw: float32 CHW normalized, size=input_size_hw
      K_new : updated intrinsics for the transformed image (in pixels of input image)
    """

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # enforce expected src size (nuScenes usually 900x1600)
    H0, W0 = img.shape[:2]
    Hs, Ws = src_size_hw
    if (H0, W0) != (Hs, Ws):
        # allow mismatch, but keep consistent by using actual size
        Hs, Ws = H0, W0

    K_new = K.astype(np.float32).copy()

    # -------- resize (test_resize=0 -> factor=1.0) --------
    # common convention: resize_factor = 1 + test_resize
    resize_factor = 1.0 + float(test_resize)
    newW = int(round(Ws * resize_factor))
    newH = int(round(Hs * resize_factor))

    if (newW, newH) != (Ws, Hs):
        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
        K_new[0, :] *= resize_factor
        K_new[1, :] *= resize_factor

    # -------- crop to input_size (center-x + bottom-y) --------
    inH, inW = input_size_hw
    if newW < inW or newH < inH:
        raise RuntimeError(
            f"After resize, image smaller than input_size: resized=({newH},{newW}) input=({inH},{inW}). "
            f"Check data_config."
        )

    # center crop x
    crop_x = int(round((newW - inW) * 0.5))
    # bottom crop y (common for driving datasets)
    crop_y = int(round(newH - inH))

    img = img[crop_y:crop_y + inH, crop_x:crop_x + inW]
    # update principal point
    K_new[0, 2] -= crop_x
    K_new[1, 2] -= crop_y

    # -------- flip (test_flip=False) --------
    if test_flip:
        img = img[:, ::-1].copy()
        # cx' = (W-1) - cx
        K_new[0, 2] = (inW - 1) - K_new[0, 2]

    # -------- rotate (test_rotate=0) --------
    if abs(test_rotate_deg) > 1e-6:
        # rotate around image center
        cx, cy = (inW - 1) * 0.5, (inH - 1) * 0.5
        M = cv2.getRotationMatrix2D((cx, cy), test_rotate_deg, 1.0)
        img = cv2.warpAffine(img, M, (inW, inH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        # update K with rotation in pixel space:
        # [u v 1]^T = A * [u' v' 1]^T ; we need equivalent update
        # easiest: convert to homography H(3x3)
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = M.astype(np.float32)
        # K' = H * K (approx for pixel transform)
        K_new = H @ K_new

    # -------- pad (your config pad=(0,0,0,0)) --------
    pl, pt, pr, pb = pad
    if any(x != 0 for x in pad):
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        K_new[0, 2] += pl
        K_new[1, 2] += pt

    # normalize -> CHW float32
    img = img.astype(np.float32)
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    return img, K_new


# =========================================================
# Build lidar->cam and projection P = K * [R|t]
# infos.pkl provides:
#   sensor2lidar_rotation: 3x3
#   sensor2lidar_translation: 3
# =========================================================
def invert_rigid(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given x_dst = R*x_src + t, return inverse: x_src = Rinv*x_dst + tinv."""
    R = np.asarray(R, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32).reshape(3,)
    Rinv = R.T
    tinv = -Rinv @ t
    return Rinv, tinv


def build_lidar2cam_from_sensor2lidar(cam: dict) -> Tuple[np.ndarray, np.ndarray]:
    R_c2l = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float32)
    t_c2l = np.asarray(cam["sensor2lidar_translation"], dtype=np.float32).reshape(3,)

    if R_c2l.shape == (3, 3):
        # ok
        pass
    else:
        raise RuntimeError(f"sensor2lidar_rotation shape unexpected: {R_c2l.shape} (expect 3x3)")

    # invert cam->lidar to lidar->cam
    R_l2c, t_l2c = invert_rigid(R_c2l, t_c2l)
    return R_l2c, t_l2c


def build_P_list_for_frame(
    cams: Dict[str, dict],
    cam_order: List[str],
    K_list_new: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Returns list length nv, each P is 3x4 (float32) mapping lidar->image pixels in the *input image*.
    """
    P_list = []
    for i, cam_name in enumerate(cam_order):
        cd = cams[cam_name]
        R_l2c, t_l2c = build_lidar2cam_from_sensor2lidar(cd)
        Rt = np.concatenate([R_l2c, t_l2c.reshape(3, 1)], axis=1)  # 3x4
        P = (K_list_new[i].astype(np.float32) @ Rt.astype(np.float32)).astype(np.float32)
        P_list.append(P)
    return P_list


# =========================================================
# Multi-frame extraction: build N=frames*nv images + per-image P
# If sweeps do not contain cams, fallback to replicate current frame.
# =========================================================
CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]

def get_frame_infos(info: dict, frames: int) -> List[dict]:
    out = [info]
    sweeps = info.get("sweeps", [])
    if not isinstance(sweeps, list):
        sweeps = []

    # sweeps may be lidar sweeps OR camera sweeps. We only accept those that have cams.
    cam_sweeps = []
    for s in sweeps:
        if isinstance(s, dict) and ("cams" in s) and isinstance(s["cams"], dict):
            cam_sweeps.append(s)

    for i in range(frames - 1):
        if i < len(cam_sweeps):
            out.append(cam_sweeps[i])
        else:
            # fallback replicate current
            out.append(info)
    return out


def build_sequence_from_infos(
    info: dict,
    data_root: str,
    frames: int,
    nv: int,
    src_size_hw: Tuple[int, int],
    input_size_hw: Tuple[int, int],
    test_resize: float,
    test_flip: bool,
    test_rotate: float,
    pad: Tuple[int, int, int, int],
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """
    Returns:
      img_paths: length N=frames*nv
      P_list   : length N each is 3x4 lidar->img projection in input image pixel space
      K_new_list: length N each is 3x3 intrinsic after preprocessing
    """
    assert nv == 6, "this script assumes nv=6 nuScenes cameras"
    frame_infos = get_frame_infos(info, frames=frames)

    img_paths: List[str] = []
    P_list: List[np.ndarray] = []
    K_new_all: List[np.ndarray] = []

    for fidx, fi in enumerate(frame_infos):
        cams = fi["cams"]

        # build per-cam paths and K updates
        paths_f = []
        K_f_new = []

        for cam_name in CAM_ORDER:
            cd = cams[cam_name]
            p = cd.get("data_path", None)
            p = normalize_path(data_root, p)
            paths_f.append(p)

        # K from infos (camera_intrinsics)
        K_raw = []
        for cam_name in CAM_ORDER:
            cd = cams[cam_name]
            K_raw.append(np.asarray(cd["camera_intrinsics"], dtype=np.float32))

        # preprocess each image to input_size and update K accordingly
        for i in range(nv):
            # we only need K_new; the actual image preprocessing will be done later again
            # but we compute K_new here for P. To avoid double-reading, we will also preprocess images later
            # (still ok). If you care about speed, cache the processed images.
            img_dummy, K_new = preprocess_image_and_K_test(
                img_path=paths_f[i],
                K=K_raw[i],
                src_size_hw=src_size_hw,
                input_size_hw=input_size_hw,
                test_resize=test_resize,
                test_flip=test_flip,
                test_rotate_deg=test_rotate,
                pad=pad,
                to_rgb=True,
            )
            # discard img_dummy to save memory, keep K_new only
            del img_dummy
            K_f_new.append(K_new)

        # P for this frame
        P_f = build_P_list_for_frame(cams, CAM_ORDER, K_f_new)

        # append to sequence
        img_paths.extend(paths_f)
        P_list.extend(P_f)
        K_new_all.extend(K_f_new)

    return img_paths, P_list, K_new_all


# =========================================================
# Backproject (aligned with model n_voxels + voxel_size + point_cloud_range)
# config:
#   n_voxels = [250,250,6]
#   voxel_size = [0.4,0.4,1.0]
#   point_cloud_range = [-50,-50,-5, 50,50,3]
# We'll set origin = pc_range_min + 0.5*(n_voxels*voxel_size)
# =========================================================
@torch.no_grad()
def make_points_xyz(
    n_voxels_xyz: Tuple[int, int, int],
    voxel_size_xyz: Tuple[float, float, float],
    pc_range: Tuple[float, float, float, float, float, float],
    device: torch.device,
) -> torch.Tensor:
    nx, ny, nz = n_voxels_xyz
    vx, vy, vz = voxel_size_xyz
    x_min, y_min, z_min, _, _, _ = pc_range

    extent = torch.tensor([nx * vx, ny * vy, nz * vz], dtype=torch.float32, device=device)
    origin = torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device) + 0.5 * extent

    n_vox = torch.tensor([nx, ny, nz], dtype=torch.float32, device=device)
    vsize = torch.tensor([vx, vy, vz], dtype=torch.float32, device=device)

    try:
        grids = torch.meshgrid(
            torch.arange(nx, device=device),
            torch.arange(ny, device=device),
            torch.arange(nz, device=device),
            indexing="ij",
        )
    except TypeError:
        grids = torch.meshgrid(
            torch.arange(nx, device=device),
            torch.arange(ny, device=device),
            torch.arange(nz, device=device),
        )
    pts = torch.stack(grids)  # [3,nx,ny,nz]
    new_origin = origin - n_vox / 2.0 * vsize
    pts = pts * vsize.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return pts  # [3,nx,ny,nz]


def backproject_one_frame(
    feat_nvchw: torch.Tensor,     # [nv,C,Hf,Wf]
    points: torch.Tensor,         # [3,nx,ny,nz]
    P_nv: torch.Tensor,           # [nv,3,4] lidar->img pixels in input image
    stride: int,
) -> torch.Tensor:
    """
    Returns: [C,nx,ny,nz]
    """
    nv, C, Hf, Wf = feat_nvchw.shape
    nx, ny, nz = points.shape[-3:]

    # scale P to feature map coords
    P = P_nv.clone()
    P[:, 0, :] /= float(stride)
    P[:, 1, :] /= float(stride)

    # [3,nx,ny,nz] -> [nv,4,N]
    pts = points.view(1, 3, -1).expand(nv, 3, -1)
    pts = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)  # [nv,4,N]

    pts2d3 = torch.bmm(P, pts)  # [nv,3,N]
    z = pts2d3[:, 2]
    x = (pts2d3[:, 0] / z).round().long()
    y = (pts2d3[:, 1] / z).round().long()

    valid = (z > 1e-5) & (x >= 0) & (y >= 0) & (x < Wf) & (y < Hf)

    volume = torch.zeros((C, pts.shape[-1]), device=feat_nvchw.device, dtype=feat_nvchw.dtype)
    for i in range(nv):
        vi = valid[i]
        if vi.any():
            volume[:, vi] = feat_nvchw[i, :, y[i, vi], x[i, vi]]

    return volume.view(C, nx, ny, nz)


def build_mlvl_volume_from_pre(
    mlvl_feat: torch.Tensor,   # [N,C,Hf,Wf] where N=frames*nv
    P_list: List[np.ndarray],  # length N, each 3x4
    nv: int,
    frames: int,
    points: torch.Tensor,
    stride: int,
    use_frames_concat: bool,
) -> torch.Tensor:
    """
    Return [1, Cvol, X, Y, Z]
    """
    device = mlvl_feat.device
    N, C, Hf, Wf = mlvl_feat.shape
    assert N == frames * nv, f"N mismatch: N={N}, frames*nv={frames*nv}"

    feat = mlvl_feat.view(frames, nv, C, Hf, Wf)  # [T,nv,C,Hf,Wf]
    P_all = np.stack(P_list, axis=0).astype(np.float32)          # [N,3,4]
    P_all = torch.from_numpy(P_all).to(device).view(frames, nv, 3, 4)

    vols = []
    frame_ids = range(frames) if use_frames_concat else [frames - 1]
    for t in frame_ids:
        vol = backproject_one_frame(feat[t], points, P_all[t], stride=stride)  # [C,X,Y,Z]
        vols.append(vol)

    vol_cat = torch.cat(vols, dim=0)  # [Cvol,X,Y,Z]
    return vol_cat.unsqueeze(0)       # [1,Cvol,X,Y,Z]


# =========================================================
# Dir correction (from your config)
# dir_offset=0.7854, dir_limit_offset=0
# =========================================================
def limit_period(val: torch.Tensor, offset: float = 0.5, period: float = math.pi) -> torch.Tensor:
    return val - torch.floor(val / period + offset) * period


def apply_dir_correction(
    bboxes: torch.Tensor,     # [K,9]
    dir_cls: torch.Tensor,    # [K] int
    dir_offset: float,
    dir_limit_offset: float
) -> torch.Tensor:
    yaw = bboxes[:, 6]
    yaw = limit_period(yaw - dir_offset, dir_limit_offset, math.pi)
    yaw = yaw + dir_offset + math.pi * dir_cls.to(yaw.dtype)
    bboxes[:, 6] = yaw
    return bboxes


# =========================================================
# Visualization
# =========================================================
def box_corners_lidar(x, y, z, w, l, h, yaw) -> np.ndarray:
    """
    lidar coord: x forward, y left, z up
    box center (x,y,z), dims (w,l,h), yaw around z.
    Returns 8 corners [8,3]
    """
    # l along x, w along y (common for nuScenes LiDAR boxes)
    dx = l / 2.0
    dy = w / 2.0
    dz = h / 2.0

    corners = np.array([
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz],
    ], dtype=np.float32)

    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    corners = corners @ R.T
    corners += np.array([x, y, z], dtype=np.float32)
    return corners


def draw_bev(
    out_path: str,
    boxes: np.ndarray,    # [M,9]
    scores: np.ndarray,   # [M]
    labels: np.ndarray,   # [M]
    pc_range: Tuple[float, float, float, float, float, float],
    score_thr: float = 0.2,
    W: int = 900,
    H: int = 900,
):
    x_min, y_min, _, x_max, y_max, _ = pc_range
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    def xy_to_uv(x, y):
        u = int((x - x_min) / (x_max - x_min) * (W - 1))
        v = int((y_max - y) / (y_max - y_min) * (H - 1))
        return u, v

    for i in range(boxes.shape[0]):
        if scores[i] < score_thr:
            continue
        x, y, z, w, l, h, yaw, vx, vy = boxes[i]
        corners = box_corners_lidar(x, y, z, w, l, h, yaw)
        # top face (0..3)
        pts = corners[:4, :2]  # xy
        poly = []
        for px, py in pts:
            u, v = xy_to_uv(px, py)
            poly.append([u, v])
        poly = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)

        cv2.polylines(canvas, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        u0, v0 = xy_to_uv(x, y)
        cv2.putText(canvas, f"{int(labels[i])}:{scores[i]:.2f}", (u0, v0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(out_path, canvas)
    print(f"[VIS] saved BEV: {out_path}")


def project_points(P: np.ndarray, pts_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    P: 3x4 lidar->img pixels
    pts_xyz: Nx3
    Returns:
      uv: Nx2
      depth: N
    """
    N = pts_xyz.shape[0]
    pts_h = np.concatenate([pts_xyz, np.ones((N, 1), dtype=np.float32)], axis=1)  # Nx4
    q = (P @ pts_h.T).T  # Nx3
    depth = q[:, 2]
    uv = q[:, :2] / np.clip(depth[:, None], 1e-6, None)
    return uv, depth


def draw_3d_boxes_on_image(
    img_bgr: np.ndarray,
    P: np.ndarray,
    boxes: np.ndarray,   # [M,9]
    scores: np.ndarray,
    labels: np.ndarray,
    score_thr: float,
) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    for i in range(boxes.shape[0]):
        if scores[i] < score_thr:
            continue
        x, y, z, w, l, h, yaw, vx, vy = boxes[i]
        corners = box_corners_lidar(x, y, z, w, l, h, yaw)  # 8x3
        uv, depth = project_points(P, corners)

        if np.all(depth <= 0):
            continue

        uv_int = uv.astype(np.int32)
        # draw edges if both ends visible
        for a, b in edges:
            if depth[a] > 0 and depth[b] > 0:
                ua, va = uv_int[a]
                ub, vb = uv_int[b]
                if (0 <= ua < W and 0 <= va < H) or (0 <= ub < W and 0 <= vb < H):
                    cv2.line(out, (ua, va), (ub, vb), (0, 255, 0), 2)

        # put label near corner 0
        u0, v0 = uv_int[0]
        if 0 <= u0 < W and 0 <= v0 < H:
            cv2.putText(out, f"{int(labels[i])}:{scores[i]:.2f}",
                        (u0, v0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return out


# =========================================================
# Main
# =========================================================
def main():
    ensure_numpy_1x()

    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pre_engine", required=True)
    ap.add_argument("--post_engine", required=True)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--nv", type=int, default=6)
    ap.add_argument("--frames", type=int, default=4)

    ap.add_argument("--src_h", type=int, default=900)
    ap.add_argument("--src_w", type=int, default=1600)
    ap.add_argument("--img_h", type=int, default=512)
    ap.add_argument("--img_w", type=int, default=1408)

    # from your config
    ap.add_argument("--n_voxels", type=int, nargs=3, default=[250, 250, 6])
    ap.add_argument("--voxel_size", type=float, nargs=3, default=[0.4, 0.4, 1.0])
    ap.add_argument("--pc_range", type=float, nargs=6, default=[-50, -50, -5, 50, 50, 3])

    # test aug from your data_config
    ap.add_argument("--test_resize", type=float, default=0.0)
    ap.add_argument("--test_rotate", type=float, default=0.0)
    ap.add_argument("--test_flip", action="store_true")  # config is False, keep default False
    ap.add_argument("--pad", type=int, nargs=4, default=[0, 0, 0, 0])  # left, top, right, bottom

    ap.add_argument("--score_thr", type=float, default=0.2)
    ap.add_argument("--dir_correction", action="store_true")
    ap.add_argument("--dir_offset", type=float, default=0.7854)
    ap.add_argument("--dir_limit_offset", type=float, default=0.0)

    ap.add_argument("--out_dir", default="./vis_out")
    ap.add_argument("--topk_vis", type=int, default=50)

    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    mkdir_p(args.out_dir)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # load engines
    pre = TRTRunnerTRT10(args.pre_engine, device=args.device, log_level=trt.Logger.INFO)
    post = TRTRunnerTRT10(args.post_engine, device=args.device, log_level=trt.Logger.INFO)

    pre_in = pre.input_names[0]
    pre_in_shape = tuple(int(x) for x in pre.engine.get_tensor_shape(pre_in))
    N_expected = int(pre_in_shape[1])
    print(f"[Engine] pre expects N={N_expected} images (shape={pre_in_shape})")

    post_in = post.input_names[0]
    post_in_shape = tuple(int(x) for x in post.engine.get_tensor_shape(post_in))
    Cvol_expected = int(post_in_shape[1])
    print(f"[Engine] post expects Cvol={Cvol_expected} (shape={post_in_shape})")

    # infer frames from engine if user provided mismatched
    if args.frames * args.nv != N_expected:
        print(f"[WARN] args.frames*nv={args.frames*args.nv} != engine N={N_expected}, override frames={N_expected//args.nv}")
        args.frames = N_expected // args.nv

    # load infos
    infos = load_infos(args.infos)
    if args.sample_idx >= len(infos):
        raise IndexError(f"sample_idx {args.sample_idx} out of range: len={len(infos)}")
    info = infos[args.sample_idx]

    # build image paths + projection matrices aligned with test aug
    img_paths, P_list, K_new_list = build_sequence_from_infos(
        info=info,
        data_root=args.data_root,
        frames=args.frames,
        nv=args.nv,
        src_size_hw=(args.src_h, args.src_w),
        input_size_hw=(args.img_h, args.img_w),
        test_resize=args.test_resize,
        test_flip=args.test_flip,
        test_rotate=args.test_rotate,
        pad=tuple(args.pad),
    )

    assert len(img_paths) == N_expected, f"img_paths len {len(img_paths)} != N_expected {N_expected}"
    print(f"[Infos] built N={len(img_paths)} img_paths (frames={args.frames}, nv={args.nv})")

    # check exist quickly
    missing = [p for p in img_paths if not os.path.exists(p)]
    if missing:
        print("[ERROR] some image paths missing, show first 5:")
        for p in missing[:5]:
            print("  ", p)
        raise FileNotFoundError("Image paths missing. Check --data_root or normalize_path rules.")

    if args.analyze_only:
        print("\n=== Analyze only ===")
        print("info keys:", list(info.keys()))
        print("cams keys:", list(info["cams"].keys()))
        ex = info["cams"][CAM_ORDER[0]]
        print("example cam keys:", list(ex.keys()))
        print("camera_intrinsics shape:", np.asarray(ex["camera_intrinsics"]).shape)
        print("sensor2lidar_rotation shape:", np.asarray(ex["sensor2lidar_rotation"]).shape)
        print("P[0] shape:", P_list[0].shape, "first_row:", P_list[0][0])
        print("=== Analyze done ===")
        return

    # build input tensor [1,N,3,H,W] with aligned preprocessing
    imgs = np.zeros((1, N_expected, 3, args.img_h, args.img_w), dtype=np.float32)

    # also keep original input images for visualization (current frame only)
    # current frame = last frame chunk (fidx = frames-1) => indices [(frames-1)*nv : frames*nv]
    cur_start = (args.frames - 1) * args.nv
    cur_end = cur_start + args.nv
    cur_imgs_rgb = []

    print("[Preprocess] reading & preprocessing images ...")
    for i in range(N_expected):
        # K_new already computed above; but we re-run preprocessing to produce pixels exactly same way
        # (same deterministic test aug)
        # NOTE: we don't need K returned now
        img_chw, _ = preprocess_image_and_K_test(
            img_path=img_paths[i],
            K=K_new_list[i],  # pass K_new as dummy; we ignore the returned K
            src_size_hw=(args.src_h, args.src_w),
            input_size_hw=(args.img_h, args.img_w),
            test_resize=0.0,           # IMPORTANT: already baked into K_new_list, so keep 0 here
            test_flip=False,
            test_rotate_deg=0.0,
            pad=(0, 0, 0, 0),
            to_rgb=True,
        )
        imgs[0, i] = img_chw

        if cur_start <= i < cur_end:
            # for vis, we want the transformed RGB image (before normalize)
            rgb = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            # apply same transform path to get final input image for drawing
            K_raw = np.asarray(info["cams"][CAM_ORDER[i-cur_start]]["camera_intrinsics"], dtype=np.float32)
            rgb_chw, _ = preprocess_image_and_K_test(
                img_path=img_paths[i],
                K=K_raw,
                src_size_hw=(args.src_h, args.src_w),
                input_size_hw=(args.img_h, args.img_w),
                test_resize=args.test_resize,
                test_flip=args.test_flip,
                test_rotate_deg=args.test_rotate,
                pad=tuple(args.pad),
                to_rgb=True,
            )
            rgb_hw = rgb_chw.transpose(1, 2, 0)
            # de-normalize for viewing
            rgb_hw = (rgb_hw * STD + MEAN).clip(0, 255).astype(np.uint8)
            cur_imgs_rgb.append(rgb_hw)

    image_input = torch.from_numpy(imgs).to(device)

    # run pre
    torch.cuda.synchronize()
    t0 = time.time()
    print("[TRT] running pre.engine ...")
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        pre_out = pre.infer({pre_in: image_input})
    torch.cuda.synchronize()
    t1 = time.time()

    pre_out_name = pre.output_names[0]
    mlvl_feat = pre_out[pre_out_name]  # [N,C,Hf,Wf]
    print(f"[Pre] {pre_out_name}: {tuple(mlvl_feat.shape)} mean={float(mlvl_feat.mean()):.6f} time={t1-t0:.3f}s")

    # build points
    points = make_points_xyz(
        n_voxels_xyz=tuple(args.n_voxels),
        voxel_size_xyz=tuple(args.voxel_size),
        pc_range=tuple(args.pc_range),
        device=device
    )

    # determine stride from feature map
    # input width / Wf
    _, C, Hf, Wf = mlvl_feat.shape
    stride = int(round(args.img_w / Wf))
    if stride <= 0:
        stride = 4
    print(f"[Mid] feature Hf,Wf=({Hf},{Wf}) => stride≈{stride}")

    # decide frames concat by post Cvol
    frames = args.frames
    use_frames_concat = True
    if Cvol_expected == frames * C:
        use_frames_concat = True
    elif Cvol_expected == C:
        use_frames_concat = False
        print("[Mid] post expects Cvol=Cfeat -> only last frame used")
    else:
        raise RuntimeError(f"Cvol mismatch: post expects {Cvol_expected}, but frames*C={frames*C}, C={C}")

    # backproject
    torch.cuda.synchronize()
    t2 = time.time()
    print("[Mid] building mlvl_volume via backproject ... (this is the heavy step)")
    mlvl_volume = build_mlvl_volume_from_pre(
        mlvl_feat=mlvl_feat,
        P_list=P_list,
        nv=args.nv,
        frames=args.frames,
        points=points,
        stride=stride,
        use_frames_concat=use_frames_concat,
    )
    torch.cuda.synchronize()
    t3 = time.time()
    print(f"[Mid] mlvl_volume: {tuple(mlvl_volume.shape)} mean={float(mlvl_volume.mean()):.6f} time={t3-t2:.3f}s")

    # run post
    torch.cuda.synchronize()
    t4 = time.time()
    print("[TRT] running post.engine ...")
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        post_out = post.infer({post_in: mlvl_volume})
    torch.cuda.synchronize()
    t5 = time.time()
    print(f"[Post] time={t5-t4:.3f}s outputs:")
    for k, v in post_out.items():
        print(f"  - {k:12s}: {tuple(v.shape)} {v.dtype} mean={float(v.float().mean()):.6f}")

    # pick outputs
    def pick(cands, default_idx):
        for n in cands:
            if n in post_out:
                return post_out[n]
        return post_out[post.output_names[default_idx]]

    scores = pick(["scores", "cls_score"], 0).float()    # [1000,10]
    bboxes = pick(["bboxes", "boxes"], 1).float()        # [1000,9]
    dir_cls = pick(["dir_cls", "dir_scores"], 2)         # [1000]

    # decode selection
    max_scores, labels = scores.max(dim=1)
    keep = max_scores > float(args.score_thr)

    max_scores = max_scores[keep]
    labels = labels[keep].long()
    bboxes = bboxes[keep]
    dir_cls = dir_cls[keep].long()

    if args.dir_correction:
        bboxes = apply_dir_correction(bboxes, dir_cls, args.dir_offset, args.dir_limit_offset)

    if max_scores.numel() == 0:
        print(f"[Result] No boxes above score_thr={args.score_thr}")
        return

    order = torch.argsort(max_scores, descending=True)
    max_scores = max_scores[order]
    labels = labels[order]
    bboxes = bboxes[order]
    dir_cls = dir_cls[order]

    topn = min(args.topk_vis, max_scores.numel())
    print(f"\n[Result] top{topn} (score_thr={args.score_thr})")
    for i in range(topn):
        box = bboxes[i].detach().cpu().numpy()
        print(f"  #{i:02d} score={float(max_scores[i]):.3f} cls={int(labels[i])} dir={int(dir_cls[i])} box={box}")

    # =======================
    # Visualization outputs
    # =======================
    boxes_np = bboxes[:topn].detach().cpu().numpy()
    scores_np = max_scores[:topn].detach().cpu().numpy()
    labels_np = labels[:topn].detach().cpu().numpy()

    # BEV
    bev_path = os.path.join(args.out_dir, f"bev_sample{args.sample_idx}.png")
    draw_bev(
        out_path=bev_path,
        boxes=boxes_np,
        scores=scores_np,
        labels=labels_np,
        pc_range=tuple(args.pc_range),
        score_thr=args.score_thr,
        W=900,
        H=900,
    )

    # Camera projections on current frame images
    # current frame P are indices cur_start..cur_end-1
    for ci in range(args.nv):
        P = P_list[cur_start + ci]  # 3x4 lidar->img pixels for that current-frame camera
        img_rgb = cur_imgs_rgb[ci]  # HxWx3 RGB
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        painted = draw_3d_boxes_on_image(img_bgr, P, boxes_np, scores_np, labels_np, args.score_thr)
        outp = os.path.join(args.out_dir, f"cam{ci}_{CAM_ORDER[ci]}_sample{args.sample_idx}.jpg")
        cv2.imwrite(outp, painted)
        print(f"[VIS] saved CAM: {outp}")

    print(f"\nDone. timing: pre={t1-t0:.3f}s mid={t3-t2:.3f}s post={t5-t4:.3f}s total={(t5-t0):.3f}s")
    print("NOTE: This is TopK+decode only (no rotated NMS). For eval-grade results you still need rotate-NMS.")


if __name__ == "__main__":
    main()

