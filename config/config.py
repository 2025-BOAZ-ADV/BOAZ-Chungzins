"""
이 파일은 더 이상 사용되지 않습니다.
새로운 설정 시스템을 위해 base_config.py를 참조하세요.
"""

# 이전 버전과의 호환성을 위해 임시로 유지
from .base_config import BaseConfig
Config = BaseConfig  # alias for backward compatibility

# MoCo parameters
K = 512  # queue size
m = 0.999  # momentum of updating key encoder
T = 0.07  # softmax temperature

# Training parameters
batch_size = 128
num_workers = 4
epochs = 300
learning_rate = 0.03
momentum = 0.9
weight_decay = 0.01

# Model parameters
dim_mlp = 128
top_k = 10
lambda_bce = 0.5
warmup_epochs = 10
