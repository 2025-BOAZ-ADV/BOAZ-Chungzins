import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config

class MoCo(nn.Module):
    def __init__(self, base_encoder):
        """MoCo v2 모델 초기화
        Args:
            base_encoder (callable): 백본 네트워크 생성 함수
        """
        super().__init__()
        self.K = Config.K
        self.m = Config.m
        self.T = Config.T
        self.top_k = Config.top_k
        self.lambda_bce = Config.lambda_bce

        # 인코더 생성
        self.encoder_q = base_encoder()  # query encoder
        self.encoder_k = base_encoder()  # key encoder

        # projection head 생성
        dim_enc = 2048  # ResNet50의 출력 차원
        dim_prj = Config.dim_mlp  # projection head의 출력 차원

        self.proj_head_q = nn.Sequential(
            nn.Linear(dim_enc, dim_enc),
            nn.BatchNorm1d(dim_enc),
            nn.GELU(),
            nn.Linear(dim_enc, dim_prj)
        )
        self.proj_head_k = nn.Sequential(
            nn.Linear(dim_enc, dim_enc),
            nn.BatchNorm1d(dim_enc),
            nn.GELU(),
            nn.Linear(dim_enc, dim_prj)
        )

        # key encoder의 파라미터를 query encoder와 동일하게 초기화하고 고정
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # queue 초기화
        self.register_buffer("queue_g", F.normalize(torch.randn(dim_enc, self.K), dim=0))
        self.register_buffer("queue_z", F.normalize(torch.randn(dim_prj, self.K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """momentum encoder 업데이트"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, g2, z2):
        """queue 업데이트"""
        batch_size = g2.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # queue에 새로운 features 추가
        self.queue_g[:, ptr:ptr + batch_size] = g2.T.detach()
        self.queue_z[:, ptr:ptr + batch_size] = z2.T.detach()
        
        # pointer 업데이트
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, im_q, im_k, epoch=None):
        """
        Args:
            im_q (Tensor): query 이미지 배치
            im_k (Tensor): key 이미지 배치
            epoch (int, optional): 현재 epoch (warmup에 사용)
        Returns:
            tuple: (total loss, logits, labels)
        """
        # encoder_q → g1 (feature)
        g1 = F.normalize(self.encoder_q(im_q), dim=1)  # shape: [B, 2048]

        # projection head → z1
        z1 = F.normalize(self.proj_head_q(g1), dim=1)  # shape: [B, 128]

        # encoder_k → g2, z2 (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            g2 = F.normalize(self.encoder_k(im_k), dim=1)
            z2 = F.normalize(self.proj_head_k(g2), dim=1)

        # top-k mining
        sim_g = torch.matmul(g1, self.queue_g.clone().detach())  # [N, K]
        topk_idx = torch.topk(sim_g, self.top_k, dim=1).indices
        y = torch.zeros_like(sim_g)
        y.scatter_(1, topk_idx, 1.0)

        # Binary Cross Entropy loss
        sim_z = torch.matmul(z1, self.queue_z.clone().detach())
        bce_loss = F.binary_cross_entropy_with_logits(sim_z / self.T, y)

        # InfoNCE loss
        l_pos = torch.sum(z1 * z2, dim=1, keepdim=True)
        l_neg = torch.matmul(z1, self.queue_z.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        info_nce_loss = F.cross_entropy(logits, labels)

        # Total loss (with optional warmup)
        if epoch is not None and epoch < Config.warmup_epochs:
            loss = info_nce_loss
        else:
            loss = info_nce_loss + self.lambda_bce * bce_loss

        # queue 업데이트
        self._dequeue_and_enqueue(g2, z2)

        return loss, logits, labels
