import argparse
import os
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

class FastBEVPre(nn.Module):
    def __init__(self, model):
        super(FastBEVPre, self).__init__()
        self.model = model

    def forward(self, imgs):
        """
        :param imgs: input tensor of shape [B, N, 3, H, W]
        :return: image features tensor
        """
        img_feats = self.model.extract_img_feat(imgs=imgs, img_metas=None)
        return img_feats


class FastBEVPost(nn.Module):
    def __init__(self, model):
        super(FastBEVPost, self).__init__()
        self.model = model

    def forward(self, img_feats, rots, trans, intrins, post_rots, post_trans):
        """
        :param img_feats: image feature tensor
        :param rots, trans, intrins, post_rots, post_trans: geometric tensors for projection
        :return: head output tensor (detection results)
        """
        bev = self.model.view_transform(img_feats, rots, trans, intrins, post_rots, post_trans)
        bev = self.model.bev_encoder(bev)
        head_out = self.model.bbox_head(bev)
        return head_out


def export_onnx(model, pre_post_model, imgs, rots, trans, intrins, post_rots, post_trans, output_dir, opset_version):
    """
    Export the pre and post models to ONNX format.

    :param model: Loaded model
    :param pre_post_model: Pre or Post model
    :param imgs: Image input tensor
    :param rots, trans, intrins, post_rots, post_trans: Geometry-related tensors
    :param output_dir: Directory to save the ONNX files
    :param opset_version: ONNX opset version
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Export pre model (backbone feature extractor)
        torch.onnx.export(
            pre_post_model[0], (imgs,),
            os.path.join(output_dir, "fastbev_pre.onnx"),
            opset_version=opset_version,
            input_names=["imgs"],
            output_names=["img_feats"],
            do_constant_folding=True
        )

        # Forward pass to get the features for post model
        img_feats = pre_post_model[0](imgs)

        # Export post model (view transformation, BEV encoding, head)
        torch.onnx.export(
            pre_post_model[1], (img_feats, rots, trans, intrins, post_rots, post_trans),
            os.path.join(output_dir, "fastbev_post.onnx"),
            opset_version=opset_version,
            input_names=["img_feats", "rots", "trans", "intrins", "post_rots", "post_trans"],
            output_names=["head_out"],
            do_constant_folding=True
        )

        print("Exported Pre and Post models to ONNX.")
    except Exception as e:
        print(f"Error during ONNX export: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--out-dir", default="onnx_out", help="Directory to save the ONNX models")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for input")
    parser.add_argument("--img-size", type=int, nargs=2, default=[512, 1408], help="Image input size [H, W]")
    args = parser.parse_args()

    # Load the model configuration and checkpoint
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    _ = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.cuda().eval()

    # Prepare model for pre/post export
    pre_model = FastBEVPre(model).cuda().eval()
    post_model = FastBEVPost(model).cuda().eval()

    # Prepare dummy input tensors based on provided configuration
    B, N, H, W = args.batch_size, 6, args.img_size[0], args.img_size[1]
    imgs = torch.randn(B, N, 3, H, W, device="cuda")

    # Create dummy geometric tensors (rotation, translation, intrinsic, etc.)
    rots = torch.randn(B, N, 3, 3, device="cuda")
    trans = torch.randn(B, N, 3, device="cuda")
    intrins = torch.randn(B, N, 3, 3, device="cuda")
    post_rots = torch.randn(B, N, 3, 3, device="cuda")
    post_trans = torch.randn(B, N, 3, device="cuda")

    # Export ONNX models
    export_onnx(model, (pre_model, post_model), imgs, rots, trans, intrins, post_rots, post_trans, args.out_dir, args.opset)


if __name__ == "__main__":
    main()
