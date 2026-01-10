import argparse
from argparse import ArgumentParser
import math
import copy
import torch
import torch.nn as nn
import onnx
import onnxsim
from onnxsim import simplify
import mmcv
from mmcv import Config
import os
import sys
import numpy as np
from collections import OrderedDict

# ================= 1. 环境与注册 =================
# 强制替换 SyncBN 为 BatchNorm2d 以支持 ONNX 导出
torch.nn.SyncBatchNorm = torch.nn.BatchNorm2d

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 显式触发 FastBEV 注册
try:
    import mmdet3d.models.detectors.fastbev
    print("FastBEV registered successfully.")
except ImportError:
    print("Warning: Could not import FastBEV detector.")

from mmseg.ops import resize
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.apis import init_model

# ================= 2. 辅助函数 =================
def simplify_onnx(onnx_path):
    print(f"Simplifying {onnx_path}...")
    try:
        onnx_model = onnx.load(onnx_path)
        model_simp, check = simplify(onnx_model)
        assert check, "simplify onnx model fail!"
        onnx.save(model_simp, onnx_path)
        print(f"Finish simplify onnx: {onnx_path}")
    except Exception as e:
        print(f"Simplify failed: {e}")

def decode(anchors, deltas):
    # FastBEV / FreeAnchor3D 解码逻辑
    xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
    xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
    
    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha
    rg = rt + ra
    zg = zg - hg / 2
    
    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

# ================= 3. 模型包装类 =================

class TRTModel_pre(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        # 处理输入形状 [1, 6, 3, 512, 1408] -> [6, 3, 512, 1408]
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.model.backbone(img)
        mlvl_feats = self.model.neck(x)
        mlvl_feats = list(mlvl_feats)

        # 多尺度融合逻辑 (对应 m5 的 multi_scale_id=[0])
        if self.model.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.model.multi_scale_id:
                if getattr(self.model, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i], 
                            size=mlvl_feats[msid].size()[2:], 
                            mode="bilinear", 
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self.model, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_
        return mlvl_feats[0]

class TRTModel_post(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.nms_pre = 1000
        
        print("Dynamic generating anchors based on m5 config...")
        # 针对 FreeAnchor3DHead 的修复：直接从 config 获取或手动指定 m5 的 250x250
        featmap_size = [(250, 250)] 
        
        # 动态生成真实 AlignedAnchor3D
        anchors = self.model.bbox_head.anchor_generator.grid_anchors(
            featmap_size, device=device)
        
        # 固化 Anchor 到模型中，导出时变为常量
        self.register_buffer('anchors_fixed', anchors[0].contiguous()) 
        print(f"M5 Anchors encoded into ONNX. Shape: {self.anchors_fixed.shape}")

    def forward(self, mlvl_volumes):
        # 3D Neck 前向
        neck_3d_feature = self.model.neck_3d.forward(mlvl_volumes)
        # Head 前向
        cls_scores, bbox_preds, dir_cls_preds = self.model.bbox_head(neck_3d_feature)
      
        cls_score = cls_scores[0][0]
        bbox_pred = bbox_preds[0][0]
        dir_cls_pred = dir_cls_preds[0][0]
        
        # 方向分类处理
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_scores = torch.max(dir_cls_pred, dim=-1)[1]
        
        # 分数处理
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.model.bbox_head.num_classes).sigmoid()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.model.bbox_head.box_code_size)
        
        # Top-K 筛选
        max_scores, _ = cls_score.max(dim=1)
        _, topk_inds = max_scores.topk(self.nms_pre)
        
        # 获取对应 Anchor 进行解码
        selected_anchors = self.anchors_fixed[topk_inds, :]
        selected_bbox_pred = bbox_pred[topk_inds, :]
        selected_scores = cls_score[topk_inds, :]
        selected_dir_cls = dir_cls_scores[topk_inds]
        
        bboxes = decode(selected_anchors, selected_bbox_pred)
        return selected_scores, bboxes, selected_dir_cls

# ================= 4. 主函数 =================

def main():
    parser = ArgumentParser()
    # 路径根据你的真实环境设置
    parser.add_argument('--config', default="configs/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.py")
    parser.add_argument('--checkpoint', default="/home/qiu_wz/BEV_Perception/Fast-BEV-dev/pretrained_models/epoch_20.pth")
    parser.add_argument('--outfile', default='model/m5_r50_onnx/')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if not os.path.exists(args.outfile): os.makedirs(args.outfile)

    # 1. 初始化并清洗 Config 中的 SyncBN
    cfg = Config.fromfile(args.config)
    def clean_syncbn(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'type' and v == 'SyncBN': d[k] = 'BN'
                clean_syncbn(v)
        elif isinstance(d, list):
            for i in d: clean_syncbn(i)
    clean_syncbn(cfg._cfg_dict)

    # 2. 构建模型并加载权重
    print(f"Building m5 model...")
    model = init_model(cfg, checkpoint=None, device=args.device)
    
    print(f"Loading m5 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. 导出 Pre-process (512x1408)
    print("\n[Action] Exporting Pre-process ONNX...")
    dummy_image = torch.randn(1, 6, 3, 512, 1408).to(args.device)
    trt_pre = TRTModel_pre(model)
    pre_path = os.path.join(args.outfile, 'fastbev_pre_m5.onnx')
    torch.onnx.export(
        trt_pre, (dummy_image,), pre_path,
        input_names=['image'], output_names=['mlvl_feat'],
        opset_version=13, do_constant_folding=True
    )

    # 4. 导出 Post-process (250x250x6)
    print("\n[Action] Exporting Post-process ONNX with dynamic M5 anchors...")
    # m5 典型的 BEV 体积输入: [Batch, Channel, X, Y, Z]
    dummy_volume = torch.randn(1, 256, 250, 250, 6).to(args.device)
    trt_post = TRTModel_post(model, args.device)
    post_path = os.path.join(args.outfile, 'fastbev_post_m5.onnx')
    torch.onnx.export(
        trt_post, (dummy_volume,), post_path,
        input_names=['mlvl_volume'],
        output_names=["scores", "bboxes", "dir_scores"],
        opset_version=13, do_constant_folding=True
    )

    # 简化模型
    simplify_onnx(pre_path)
    simplify_onnx(post_path)
    print(f"\nSUCCESS! ONNX files for m5 generated in: {args.outfile}")

if __name__ == '__main__':
    main()