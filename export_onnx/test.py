import torch
checkpoint = torch.load("/root/autodl-tmp/Fast-BEV-dev/model/resnet18int8head/bev_ptq_head.pth", map_location="cpu")
print(type(checkpoint))
