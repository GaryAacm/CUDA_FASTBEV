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
# Utils: path resolve
# -----------------------------
def resolve_path(data_root: str, p: str) -> str:
    """
    infos.pkl 里的 data_path 常见形式：
      - "samples/CAM_FRONT/xxx.jpg"
      - "./data/nuscenes/samples/CAM_FRONT/xxx.jpg"
      - "data/nuscenes/samples/CAM_FRONT/xxx.jpg"
    你在 Orin 上 data_root 已经是 nuscenes_mini 根目录（里面有 samples/ sweeps/ v1.0-mini/）
    所以要把前面的 ./data/nuscenes/ 或 data/nuscenes/ 去掉
    """
    if p is None:
        return p
    p = p.replace("\\", "/")
    # strip known prefixes
    for pref in ["./data/nuscenes/", "data/nuscenes/", "./nuscenes/", "nuscenes/"]:
        if p.startswith(pref):
            p = p[len(pref):]
            break
    # strip leading ./ again
    if p.startswith("./"):
        p = p[2:]
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(data_root, p))


# -----------------------------
# TRT10 runner (torch buffers)
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
    def __init__(self, engine_path: str, device: str = "cuda:0", log_level=trt.Logger.ERROR):
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
# Rotation / Transform helpers
# -----------------------------
def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_to_R(rot: Any) -> np.ndarray:
    """
    支持：
      - quaternion [w,x,y,z] 或 [x,y,z,w]
      - rodrigues [rx,ry,rz]  (cv2.Rodrigues)
      - euler [roll,pitch,yaw] (fallback)
      - 3x3 or flat 9
    """
    rot = np.array(rot, dtype=np.float32).reshape(-1)

    if rot.size == 9:
        return rot.reshape(3, 3)

    if rot.size == 4:
        w, x, y, z = rot
        # 如果明显不像 wxyz，尝试 xyzw
        if abs(w) > 1.5 and abs(rot[3]) <= 1.5:
            x, y, z, w = rot

        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
        return R

    if rot.size == 3:
        rx, ry, rz = rot.tolist()

        # 优先按 Rodrigues 解释（nuScenes/mmdet3d 有时会这样存）
        R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float32))
        if R is not None and R.shape == (3, 3):
            return R.astype(np.float32)

        # fallback Euler: Rz*Ry*Rx
        cr, sr = math.cos(rx), math.sin(rx)
        cp, sp = math.cos(ry), math.sin(ry)
        cy, sy = math.cos(rz), math.sin(rz)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ], dtype=np.float32)
        return R

    raise ValueError(f"Unknown rotation format, size={rot.size}, rot={rot[:10]}")


def build_lidar2img_from_cam(cd: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (K, E)，其中：
      - K: 3x3
      - E: 3x4, lidar -> cam  (即 [R|t] in cam frame)
    已知 infos 里常见的是 sensor2lidar_* (cam -> lidar)，所以需要取逆得到 lidar -> cam
    """
    K = np.array(cd["camera_intrinsics"], dtype=np.float32)
    if K.shape == (4, 4):
        K = K[:3, :3]

    rot = cd.get("sensor2lidar_rotation", None)
    trans = cd.get("sensor2lidar_translation", None)
    if rot is None or trans is None:
        raise KeyError("cam dict missing sensor2lidar_rotation/sensor2lidar_translation")

    R_cam_lidar = rot_to_R(rot)
    t_cam_lidar = np.array(trans, dtype=np.float32).reshape(3)

    T_cam_lidar = make_T(R_cam_lidar, t_cam_lidar)
    T_lidar_cam = invert_T(T_cam_lidar)
    E = T_lidar_cam[:3, :]  # 3x4

    return K, E


def scale_K_for_stride(K: np.ndarray, stride: int) -> np.ndarray:
    K2 = K.copy().astype(np.float32)
    K2[0, :] /= float(stride)
    K2[1, :] /= float(stride)
    return K2


# -----------------------------
# Backproject
# -----------------------------
@torch.no_grad()
def get_points(n_voxels_xyz: Tuple[int, int, int], voxel_size_xyz: Tuple[float, float, float], origin_xyz: Tuple[float, float, float],
               device: torch.device) -> torch.Tensor:
    nx, ny, nz = n_voxels_xyz
    voxel_size = torch.tensor(voxel_size_xyz, dtype=torch.float32, device=device)
    n_voxels = torch.tensor([nx, ny, nz], dtype=torch.float32, device=device)
    origin = torch.tensor(origin_xyz, dtype=torch.float32, device=device)

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
    points = torch.stack(grids)  # [3,nx,ny,nz]
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_inplace(features: torch.Tensor, points: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """
    features: [nv,C,Hf,Wf]
    points: [3,X,Y,Z]
    proj: [nv,3,4]  (already includes K/stride)
    -> volume: [C,X,Y,Z]
    """
    nv, c, h, w = features.shape
    pts = points.view(1, 3, -1).expand(nv, 3, -1)
    pts = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)  # [nv,4,N]

    pts2d3 = torch.bmm(proj, pts)  # [nv,3,N]
    x = (pts2d3[:, 0] / pts2d3[:, 2]).round().long()
    y = (pts2d3[:, 1] / pts2d3[:, 2]).round().long()
    z = pts2d3[:, 2]

    valid = (x >= 0) & (y >= 0) & (x < w) & (y < h) & (z > 0)

    volume = torch.zeros((c, pts.shape[-1]), device=features.device, dtype=features.dtype)
    for i in range(nv):
        vi = valid[i]
        volume[:, vi] = features[i, :, y[i, vi], x[i, vi]]

    X, Y, Z = points.shape[-3:]
    return volume.view(c, X, Y, Z)


def build_mlvl_volume(
    image_input: torch.Tensor,       # [1,N,3,H,W]
    mlvl_feat_nchw: torch.Tensor,     # [N,C,Hf,Wf]
    P_list: List[np.ndarray],         # length N, each 3x4 (lidar2img)
    nv: int,
    frames: int,
    n_voxels_xyz: Tuple[int, int, int],
    voxel_size_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    use_frames_concat: bool,
) -> torch.Tensor:
    device = mlvl_feat_nchw.device
    N, C, Hf, Wf = mlvl_feat_nchw.shape
    assert N == frames * nv, f"N={N} != frames*nv={frames*nv}"

    stride = math.ceil(int(image_input.shape[-1]) / int(Wf))

    # points grid
    points = get_points(n_voxels_xyz, voxel_size_xyz, origin_xyz, device=device)

    feat_5d = mlvl_feat_nchw.view(frames, nv, C, Hf, Wf)  # [T,nv,C,Hf,Wf]

    # P_list -> proj_tensors per frame
    vols = []
    frame_ids = range(frames) if use_frames_concat else [frames - 1]
    for t in frame_ids:
        P_chunk = []
        for i in range(nv):
            P = P_list[t * nv + i]  # 3x4
            # stride scaling should be applied to K part; easiest: scale first 2 rows of P
            P2 = P.copy().astype(np.float32)
            P2[0, :] /= float(stride)
            P2[1, :] /= float(stride)
            P_chunk.append(P2)
        proj = torch.tensor(np.stack(P_chunk, axis=0), dtype=torch.float32, device=device)  # [nv,3,4]
        vol = backproject_inplace(feat_5d[t], points, proj)  # [C,X,Y,Z]
        vols.append(vol)

    vol_cat = torch.cat(vols, dim=0)  # [Cvol,X,Y,Z]
    return vol_cat.unsqueeze(0)       # [1,Cvol,X,Y,Z]


# -----------------------------
# Infos parsing
# -----------------------------
def load_infos(pkl_path: str) -> List[dict]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "infos" in obj:
        return obj["infos"]
    if isinstance(obj, list):
        return obj
    raise TypeError(f"Unknown infos format: {type(obj)}")


def same_scene(a: dict, b: dict) -> bool:
    for k in ["scene_token", "scene_name", "scene"]:
        if k in a and k in b:
            return a[k] == b[k]
    # fallback: use lidar_path prefix token
    return True


def pick_frame_indices(infos: List[dict], idx: int, frames: int) -> List[int]:
    """
    取 idx 为 current，向前取 frames-1 帧，尽量保持同一 scene。
    不够就 pad（重复最早可用帧）
    """
    out = [idx]
    j = idx - 1
    while len(out) < frames and j >= 0:
        if same_scene(infos[idx], infos[j]):
            out.append(j)
        else:
            break
        j -= 1
    # pad
    while len(out) < frames:
        out.append(out[-1])
    # 最终按时间从旧到新排列
    out = list(reversed(out))
    return out


def extract_queue_from_infos(
    infos: List[dict],
    sample_idx: int,
    frames: int,
    nv: int,
    data_root: str,
    cam_order: List[str],
    verbose: bool = True,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    返回：
      img_paths: length N=frames*nv
      P_list:    length N=frames*nv, each 3x4 = K @ E  (lidar->cam->img)
    """
    idxs = pick_frame_indices(infos, sample_idx, frames)
    if verbose:
        print(f"[Infos] frame indices (old->new): {idxs}")

    img_paths: List[str] = []
    P_list: List[np.ndarray] = []

    for fi, idx in enumerate(idxs):
        info = infos[idx]
        cams = info.get("cams", None)
        if cams is None:
            raise KeyError("infos item has no 'cams'")

        for cam in cam_order:
            if cam not in cams:
                raise KeyError(f"cams missing {cam}, keys={list(cams.keys())}")
            cd = cams[cam]

            # image path
            dp = cd.get("data_path", None)
            if dp is None:
                raise KeyError(f"{cam} has no data_path. keys={list(cd.keys())}")
            p = resolve_path(data_root, dp)
            img_paths.append(p)

            # K,E -> P
            K, E = build_lidar2img_from_cam(cd)
            P = (K @ E).astype(np.float32)  # 3x4
            P_list.append(P)

    assert len(img_paths) == frames * nv
    assert len(P_list) == frames * nv
    return img_paths, P_list


def analyze_one_sample(
    infos: List[dict],
    sample_idx: int,
    frames: int,
    nv: int,
    data_root: str,
    cam_order: List[str],
    max_missing_print: int = 5
) -> None:
    info = infos[sample_idx]
    print("\n=== Analyze infos[sample_idx] ===")
    print("keys:", list(info.keys()))
    cams = info.get("cams", {})
    print("cams keys:", list(cams.keys()))
    if cams:
        one = cams[cam_order[0]]
        print(f"example cam '{cam_order[0]}' keys:", list(one.keys()))
        rot = one.get("sensor2lidar_rotation", None)
        trans = one.get("sensor2lidar_translation", None)
        K = one.get("camera_intrinsics", None)
        print("sensor2lidar_rotation len:", (len(rot) if hasattr(rot, "__len__") else None), "val:", rot)
        print("sensor2lidar_translation len:", (len(trans) if hasattr(trans, "__len__") else None), "val:", trans)
        if K is not None:
            K = np.array(K)
            print("camera_intrinsics shape:", K.shape)

    img_paths, P_list = extract_queue_from_infos(
        infos, sample_idx=sample_idx, frames=frames, nv=nv,
        data_root=data_root, cam_order=cam_order, verbose=False
    )

    missing = []
    for p in img_paths:
        if not os.path.exists(p):
            missing.append(p)

    print(f"[Analyze] built N={len(img_paths)} images (frames={frames}, nv={nv})")
    if missing:
        print(f"[Analyze] MISSING {len(missing)} images! (show first {max_missing_print})")
        for p in missing[:max_missing_print]:
            print("  -", p)
        print("=> 这通常是 infos.pkl 的 data_path 前缀与 data_root 不一致。脚本已做常见前缀剥离。")
    else:
        print("[Analyze] all image paths exist ✅")

    # sanity check P
    dets = []
    for i in range(min(3, len(P_list))):
        P = P_list[i]
        print(f"[Analyze] P[{i}] shape={P.shape} dtype={P.dtype} first_row={P[0]}")
    # just check rotations det using E
    for cam in cam_order[:1]:
        cd = infos[sample_idx]["cams"][cam]
        _, E = build_lidar2img_from_cam(cd)
        R = E[:, :3]
        det = np.linalg.det(R)
        dets.append(det)
    print(f"[Analyze] det(R) example = {dets[0]:.6f} (should be near +1)")

    print("=== Analyze done ===\n")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pre_engine", required=True)
    ap.add_argument("--post_engine", required=True)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--nv", type=int, default=6)
    ap.add_argument("--frames", type=int, default=4, help="temporal frames, f4 => 4")
    ap.add_argument("--img_h", type=int, default=512)
    ap.add_argument("--img_w", type=int, default=1408)
    ap.add_argument("--n_voxels", type=int, nargs=3, default=[250, 250, 6])      # X Y Z
    ap.add_argument("--voxel_size", type=float, nargs=3, default=[0.4, 0.4, 1.5]) # 常见 m5: 250*0.4=100m => [-50,50]
    ap.add_argument("--origin", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="BEV grid center (origin). If your training used different pc_range, set accordingly.")
    ap.add_argument("--score_thr", type=float, default=0.2)
    ap.add_argument("--analyze_only", action="store_true", help="only parse+validate infos, do not run TRT")
    args = ap.parse_args()

    # numpy 版本提示（你现在已经是 1.26.4 就很好）
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError(f"NumPy is {np.__version__}, please downgrade to 1.26.x for Jetson torch build.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # Load engines first to get expected shapes
    pre = TRTRunnerTRT10(args.pre_engine, device=args.device, log_level=trt.Logger.INFO)
    post = TRTRunnerTRT10(args.post_engine, device=args.device, log_level=trt.Logger.INFO)

    # Expected N from pre engine
    pre_in = pre.input_names[0]
    pre_in_shape = tuple(int(x) for x in pre.engine.get_tensor_shape(pre_in))
    if any(d <= 0 for d in pre_in_shape):
        raise RuntimeError(f"pre engine input is dynamic: {pre_in_shape}, this script assumes fixed.")
    N_expected = int(pre_in_shape[1])
    print(f"[Engine] pre expects N={N_expected} images (shape={pre_in_shape})")

    post_in = post.input_names[0]
    post_in_shape = tuple(int(x) for x in post.engine.get_tensor_shape(post_in))
    if any(d <= 0 for d in post_in_shape):
        raise RuntimeError(f"post engine input is dynamic: {post_in_shape}, this script assumes fixed.")
    Cvol_expected = int(post_in_shape[1])
    print(f"[Engine] post expects Cvol={Cvol_expected} (shape={post_in_shape})")

    # Load infos
    infos = load_infos(args.infos)
    if args.sample_idx >= len(infos):
        raise IndexError(f"sample_idx {args.sample_idx} out of range, infos len={len(infos)}")

    cam_order = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]

    # frames/nv check with engine
    if args.frames * args.nv != N_expected:
        print(f"[WARN] frames*nv={args.frames*args.nv} != engine N_expected={N_expected}. "
              f"Will override frames to match engine: frames={N_expected//args.nv}")
        args.frames = N_expected // args.nv

    # 1) Analyze first
    analyze_one_sample(infos, args.sample_idx, args.frames, args.nv, args.data_root, cam_order)

    if args.analyze_only:
        print("[Exit] analyze_only enabled, stop before TRT inference.")
        return

    # 2) Extract queue (paths + projections)
    img_paths, P_list = extract_queue_from_infos(
        infos, sample_idx=args.sample_idx, frames=args.frames, nv=args.nv,
        data_root=args.data_root, cam_order=cam_order, verbose=True
    )

    # verify files exist (fail fast)
    for p in img_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")

    # 3) Build input tensor [1,N,3,H,W]
    N = len(img_paths)
    imgs = np.zeros((1, N, 3, args.img_h, args.img_w), dtype=np.float32)
    for i, p in enumerate(img_paths):
        imgs[0, i] = preprocess_image(p, hw=(args.img_h, args.img_w))
    image_input = torch.from_numpy(imgs).to(device)

    # 4) pre inference
    pre_out = pre.infer({pre_in: image_input})
    pre_out_name = pre.output_names[0]
    mlvl_feat = pre_out[pre_out_name]  # expected [N,C,Hf,Wf]
    print(f"[Pre] {pre_out_name}: shape={tuple(mlvl_feat.shape)} dtype={mlvl_feat.dtype} mean={float(mlvl_feat.mean()):.6f}")

    if mlvl_feat.dim() != 4 or mlvl_feat.shape[0] != N:
        raise RuntimeError(f"Unexpected pre output shape: {tuple(mlvl_feat.shape)} (expected (N,C,Hf,Wf) with N={N})")

    # 5) decide concat frames or not by post expected channel
    Cfeat = int(mlvl_feat.shape[1])
    frames = args.frames
    use_frames_concat = True
    if Cvol_expected == frames * Cfeat:
        use_frames_concat = True
    elif Cvol_expected == Cfeat:
        use_frames_concat = False
        print("[Mid] post expects Cvol=Cfeat => only use last frame in backproject")
    else:
        raise RuntimeError(f"Cvol mismatch: post expects {Cvol_expected}, but frames*Cfeat={frames*Cfeat}, Cfeat={Cfeat}")

    # 6) backproject to mlvl_volume
    mlvl_volume = build_mlvl_volume(
        image_input=image_input,
        mlvl_feat_nchw=mlvl_feat,
        P_list=P_list,
        nv=args.nv,
        frames=frames,
        n_voxels_xyz=tuple(args.n_voxels),
        voxel_size_xyz=tuple(args.voxel_size),
        origin_xyz=tuple(args.origin),
        use_frames_concat=use_frames_concat,
    )
    print(f"[Mid] mlvl_volume: shape={tuple(mlvl_volume.shape)} dtype={mlvl_volume.dtype}")

    # sanity: must match post input
    if tuple(mlvl_volume.shape) != post_in_shape:
        raise RuntimeError(f"mlvl_volume shape {tuple(mlvl_volume.shape)} != post expected {post_in_shape}")

    # 7) post inference
    post_out = post.infer({post_in: mlvl_volume})
    print("[Post] outputs:")
    for k, v in post_out.items():
        print(f"  - {k:10s} shape={tuple(v.shape)} dtype={v.dtype}")

    # 8) simple decode output view (TopK+decode already inside your post engine)
    # find outputs
    def pick(cands: List[str], fallback_idx: int) -> torch.Tensor:
        for n in cands:
            if n in post_out:
                return post_out[n]
        return post_out[post.output_names[fallback_idx]]

    scores = pick(["scores", "cls_score"], 0).float()   # [K,num_cls]
    bboxes = pick(["bboxes", "boxes"], 1).float()       # [K,9]
    dir_cls = pick(["dir_cls", "dir"], 2)

    max_scores, labels = scores.max(dim=1)
    keep = max_scores > float(args.score_thr)
    max_scores = max_scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]
    dir_cls = dir_cls[keep]

    if max_scores.numel() == 0:
        print(f"[Result] No boxes above score_thr={args.score_thr}")
        return

    order = torch.argsort(max_scores, descending=True)
    max_scores = max_scores[order]
    labels = labels[order]
    bboxes = bboxes[order]
    dir_cls = dir_cls[order]

    topn = min(20, max_scores.numel())
    print(f"\n[Result] top{topn} (score_thr={args.score_thr})")
    for i in range(topn):
        print(f"  #{i:02d} score={float(max_scores[i]):.3f} cls={int(labels[i])} dir={int(dir_cls[i])} box={bboxes[i].cpu().numpy()}")

    print("\nDone. (TopK+decode only; full pipeline usually adds rotated NMS.)")


if __name__ == "__main__":
    main()

