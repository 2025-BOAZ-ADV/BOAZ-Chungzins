import torch
import torch.nn as nn
import torch.nn.functional as F

# K: queue_g의 크기
# dim_enc: projector 통과 전, g1,g2 벡터의 차원
# dim_prj: projector 통과 후, z1,z2 벡터의 차원
class MoCo(nn.Module):
    def __init__(self, base_encoder, dim_prj=128, K=512, m=0.999, T=0.07, top_k=10, lambda_bce=0.5):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.top_k = top_k
        self.lambda_bce = lambda_bce

        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        dim_enc = 2048
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

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue_g", F.normalize(torch.randn(dim_enc, K), dim=0))      # g2를 정규화한 후 열 단위로 Qg에 저장
        self.register_buffer("queue_z", F.normalize(torch.randn(dim_prj, K), dim=0))      # z2를 정규화한 후 열 단위로 Qz에 저장
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))               # 현재 queue에 새로 쓸 위치(인덱스)를 추적하는 포인터 역할

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, g2, z2):
        batch_size = g2.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        # self.queue_g[:, ptr:ptr + batch_size] = g2.T.detach()[:, :self.queue_g.shape[1] - ptr]  # Update only available space
        # self.queue_z[:, ptr:ptr + batch_size] = z2.T.detach()[:, :self.queue_z.shape[1] - ptr]  # Update only available space
        self.queue_g[:, ptr:ptr+batch_size] = g2.T.detach()
        self.queue_z[:, ptr:ptr+batch_size] = z2.T.detach()
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, im_q, im_k, epoch=None, warmup_epochs=10):
        # encoder_q → g1 (feature)
        g1 = F.normalize(self.encoder_q(im_q), dim=1)  # shape: [B, 2048]

        # projection head → z1
        z1 = F.normalize(self.proj_head_q(g1), dim=1)  # shape: [B, 128]

        # encoder k
        with torch.no_grad():
            self._momentum_update_key_encoder()
            g2 = F.normalize(self.encoder_k(im_k), dim=1)
            z2 = F.normalize(self.proj_head_k(g2), dim=1)

        # top-k mining
        sim_g = torch.matmul(g1, self.queue_g.clone().detach())  # [N, K]
        topk_idx = torch.topk(sim_g, self.top_k, dim=1).indices
        y = torch.zeros_like(sim_g)
        y.scatter_(1, topk_idx, 1.0)

        # logits from z1 · Qz
        sim_z = torch.matmul(z1, self.queue_z.clone().detach())
        bce_loss = F.binary_cross_entropy_with_logits(sim_z / self.T, y)

        # InfoNCE loss
        l_pos = torch.sum(z1 * z2, dim=1, keepdim=True)
        l_neg = torch.matmul(z1, self.queue_z.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        info_nce_loss = F.cross_entropy(logits, labels)

        # Total loss (with optional warmup)
        if epoch is not None and epoch < warmup_epochs:
            loss = info_nce_loss
        else:
            loss = info_nce_loss + self.lambda_bce * bce_loss

        self._dequeue_and_enqueue(g2, z2)

        return loss, logits, labels