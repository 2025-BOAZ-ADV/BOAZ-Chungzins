import torch.nn.functional as F

# InfoNCE loss
def info_nce_loss(logits, labels):
    return F.cross_entropy(logits, labels)

# Binary Cross Entropy loss

# 