"""실험 001: Baseline 실험"""

import torch
import torch.nn as nn

from config.base_config import BaseConfig
from utils.logger import get_timestamp

# 사전훈련 파라미터 설정
class SSLConfig(BaseConfig):
    def __init__(self):
        # 음성 전처리 파라미터 (FinetuneConfig와 똑같이 맞춰주세요.)
        self.target_sr = 4000
        self.frame_size = 1024
        self.hop_length = 512
        self.n_mels = 128
        self.target_sec = 8

        # Augmentation 조합 리스트
        self.augmentations = [
            {
                'type': 'SpecAugment',
                'params': {
                    'time_mask_param': 0.3,
                    'freq_mask_param': 0.3
                }
            }
        ]

        # Multi-label MoCo 파라미터
        self.K = 512              # memory queue size
        self.m = 0.999            # momentum
        self.T = 0.07             # temperature
        self.dim_mlp = 128        # projector q,k의 output z1,z2의 차원
        self.top_k = 10           # positive pair의 개수
        self.lambda_bce = 0.5     # BCE loss에 곱해지는 lambda
        self.warmup_epochs = 10   # 초기에 InfoNCE loss만 사용하는 epoch

        # 훈련 파라미터
        self.batch_size = 128
        self.num_workers = 4
        self.epochs = 300
        self.learning_rate = 0.03
        self.weight_decay = 0.01

# 파인튜닝 파라미터 설정
class FinetuneConfig:
    def __init__(self):
        # 음성 전처리 파라미터 (FinetuneConfig와 똑같이 맞춰주세요.)
        self.target_sr = 4000
        self.frame_size = 1024
        self.hop_length = 512
        self.n_mels = 128
        self.target_sec = 8

        # 분류기 구조
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

        # 분류기 설정값
        self.num_layers = 3       # 분류기 구조 변경 시 직접 입력해주세요.
        self.dropout_rate = 0.5   # 분류기 구조 변경 시 직접 입력해주세요.
        self.freeze_backbone = True

        # 훈련 파라미터
        self.batch_size = 128
        self.num_workers = 4
        self.epochs = 100
        self.learning_rate = 0.03
        self.weight_decay = 0.01

# 실험 파라미터 설정
class ExperimentConfig:
    def __init__(self, exp_name: str):
        self.name = "exp001_basic_experiment"
        self.description = "Baseline 실험"
        self.exp_name = exp_name
        
        # 실험 시드
        self.seed = 42

        # 설정 인스턴스
        self.ssl = SSLConfig()
        self.finetune = FinetuneConfig()
        
        # 사전훈련 실험 이름
        self.step1_experiment_name = (
            f"{self.exp_name}-pretrain-"
            f"{self.ssl.batch_size}bs-{self.ssl.target_sr}sr-"
            f"top{self.ssl.top_k}-{self.ssl.K}K-"
            f"{self.dim_mlp}dim-{self.ssl.lambda_bce}ld-"
            f"{get_timestamp()}"
        )

        # 파인튜닝 실험 이름
        self.step2_experiment_name = (
            f"{self.exp_name}-finetune-"
            f"{self.ssl.batch_size}bs-{self.ssl.target_sr}sr-"
            f"{self.num_layers}layer-{self.dropout_rate}dr-"
            f"{get_timestamp()}"
        )

        # 성능평가 실험 이름
        self.step3_experiment_name = (
            f"{self.exp_name}-test-"
            f"{self.ssl.batch_size}bs-{self.ssl.target_sr}sr-"
            f"{self.num_layers}layer-{self.dropout_rate}dr-"
            f"{get_timestamp()}"
        )

        # 사전훈련 - 파인튜닝 train data 분할 비율
        self.split_ratio = 0.8

        # 파인튜닝 단계에서 Validation Loader 설정
        self.allow_val = False
        self.val_ratio = 0.3

        # wandb 설정
        self.wandb_project = "ICBHI_MLS_MoCo"
        self.wandb_entity = "boaz_woony-boaz"  # 이 줄은 변경하지 마세요.

        # 저장 경로
        self.checkpoint_dir = "checkpoints/exp001"
        self.log_dir = "logs/exp001"