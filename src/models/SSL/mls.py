import torch.nn as nn

class MLS(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        # 두 큐(Qg, Qz), momentum encoder 등 초기화

    def forward(self, im_q, im_k):
        # InfoNCE + BCEWithLogits 합산
        return # loss