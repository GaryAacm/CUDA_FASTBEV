import torch
import collections
from mmdet3d.ops import SparseBasicBlock
# 适配 spconv 2.x
import spconv
import spconv.pytorch as spconv_core
from spconv.pytorch import SparseSequential, SubMConv3d, SparseConv3d, SparseConvTensor
from .quantize import QuantAdd, SparseConvolutionQunat
import numpy as np

def text_format_to_color(text):
    text = text.replace("<red>", "\033[31m")
    text = text.replace("</red>", "\033[0m")
    text = text.replace("<green>", "\033[32m")
    text = text.replace("</green>", "\033[0m")
    text = text.replace("<yellow>", "\033[33m")
    text = text.replace("</yellow>", "\033[0m")
    text = text.replace("<blue>", "\033[34m")
    text = text.replace("</blue>", "\033[0m")
    text = text.replace("<mag>", "\033[35m")
    text = text.replace("</mag>", "\033[0m")
    text = text.replace("<cyan>", "\033[36m")
    text = text.replace("</cyan>", "\033[0m")
    return text

def cprint(*args, **kwargs):
    args = list(args)
    for i, item in enumerate(args):
        if isinstance(item, str):
            args[i] = text_format_to_color(item)
    print(*args, **kwargs)

def fuse_bn_weights(conv_w_OKI, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    NDim = conv_w_OKI.ndim - 2
    permute = [0, NDim+1] + [i+1 for i in range(NDim)]
    conv_w_OIK = conv_w_OKI.permute(*permute)
    # OIDHW
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w_OIK = conv_w_OIK * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w_OIK.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    permute = [0,] + [i+2 for i in range(NDim)] + [1,]
    conv_w_OKI = conv_w_OIK.permute(*permute).contiguous()
    return torch.nn.Parameter(conv_w_OKI), torch.nn.Parameter(conv_b)

def fuse_bn(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    # conv.weight 形状通常为 [K, K, K, I, O] -> 需要处理维度
    conv.weight, conv.bias = fuse_bn_weights(conv.weight.permute(4, 0, 1, 2, 3), conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    conv.weight.data = conv.weight.data.permute(1, 2, 3, 4, 0)

def load_checkpoint(model, file, startsname=None):
    device   = next(model.parameters()).device
    ckpt     = torch.load(file, map_location=device)["state_dict"]
    
    if startsname is not None:
        new_ckpt = collections.OrderedDict()
        for key, val in ckpt.items():
            if key.startswith(startsname):
                newkey = key[len(startsname)+1:]
                new_ckpt[newkey] = val
    else:
        new_ckpt = ckpt

    model.load_state_dict(new_ckpt, strict=True)

def replace_feature(self, feature: torch.Tensor):
    """适配 spconv 2.x 的 SparseConvTensor 结构"""
    new_spt = SparseConvTensor(feature, self.indices, self.spatial_shape,
                                self.batch_size)
    # 2.x 可能需要手动带入 grid 等属性，但基础构造仅需以上参数
    if hasattr(self, 'grid'):
        new_spt.grid = self.grid
    return new_spt

class new_sparse_basic_block_forward:
    def __init__(self, obj, is_fuse_relu):
        self.obj = obj
        self.is_fuse_relu = is_fuse_relu

    def __call__(self, x):
        is_fuse_relu = self.is_fuse_relu
        self = self.obj

        identity = x
        out = self.conv1(x)
        if not is_fuse_relu:
            out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        if hasattr(self, 'quant_add'):
            out = replace_feature(out, self.quant_add(out.features, identity.features))
        else:
            out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))
        return out

def fuse_sparse_basic_block(self, is_fuse_bn=False, is_fuse_relu=True):
    if not isinstance(self.forward, new_sparse_basic_block_forward):
        self.forward = new_sparse_basic_block_forward(self, is_fuse_relu)
    else:
        self.forward.is_fuse_relu = is_fuse_relu

    if is_fuse_relu == True:
        self.conv1.act_type = "ReLU"

    if is_fuse_bn == True:
        fuse_bn(self.conv1, self.bn1)
        fuse_bn(self.conv2, self.bn2)
        if hasattr(self, "bn1"): delattr(self, "bn1")
        if hasattr(self, "bn2"): delattr(self, "bn2")

def layer_fusion_bn(model):
    def set_attr_by_path(m, path, newval):
        arr = path.split(".")
        parent = m
        for i in range(len(arr) - 1):
            parent = getattr(parent, arr[i])
        setattr(parent, arr[-1], newval)

    for name, module in model.named_modules():
        if isinstance(module, SparseSequential):
            # 适配 2.x 的 SparseConvolution 类型检查
            if isinstance(module[0], (spconv_core.conv.SparseConvolution, SparseConvolutionQunat)):
                c, b, r = [module[i] for i in range(3)]
                fuse_bn(c, b)
                # 重新包装
                new_seq = SparseSequential(c, r)
                set_attr_by_path(model, name, new_seq)
        elif isinstance(module, SparseBasicBlock):
            fuse_sparse_basic_block(module, is_fuse_bn=True, is_fuse_relu=False)
        elif isinstance(module, torch.nn.ReLU): 
            module.inplace = False
    return model

def fuse_relu_only(model):
    def set_attr_by_path(m, path, newval):
        arr = path.split(".")
        parent = m
        for i in range(len(arr) - 1):
            parent = getattr(parent, arr[i])
        setattr(parent, arr[-1], newval)

    for name, module in model.named_modules():
        if isinstance(module, SparseSequential):
            if isinstance(module[0], spconv_core.conv.SparseConvolution):
                c, r = [module[i] for i in range(2)]
                c.act_type = "ReLU"
                set_attr_by_path(model, name, c)
        elif isinstance(module, SparseBasicBlock):
            fuse_sparse_basic_block(module, is_fuse_bn=False, is_fuse_relu=True)
        elif isinstance(module, torch.nn.ReLU): 
            module.inplace = False
    return model

def layer_fusion_bn_relu(model):
    def set_attr_by_path(m, path, newval):
        arr = path.split(".")
        parent = m
        for i in range(len(arr) - 1):
            parent = getattr(parent, arr[i])
        setattr(parent, arr[-1], newval)

    for name, module in model.named_modules():
        if isinstance(module, SparseSequential):
            if len(module) >= 3 and isinstance(module[0], (SubMConv3d, SparseConv3d)):
                c, b, r = [module[i] for i in range(3)]
                fuse_bn(c, b)
                c.act_type = "ReLU"
                set_attr_by_path(model, name, c)
        elif isinstance(module, SparseBasicBlock):
            fuse_sparse_basic_block(module, is_fuse_bn=True, is_fuse_relu=True)
        elif isinstance(module, torch.nn.ReLU): 
            module.inplace = False
    return model