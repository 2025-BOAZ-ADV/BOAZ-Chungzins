"""실험 001: Baseline 실험"""

import torch
import torch.nn as nn

from config.base_config import BaseConfig
from utils.logger import get_timestamp

# 사전훈련 파라미터 설정
class SSLConfig:
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
        self.warmup_epochs = 3   # 초기에 InfoNCE loss만 사용하는 epoch, default: 10

        # 훈련 파라미터
        self.batch_size = 128
        self.num_workers = 4
        self.epochs = 8   # default: 300
        self.learning_rate = 0.03
        self.weight_decay = 0.01

        # 캐시 사용 여부 (캐시 = ICBHI train data의 각 cycle의 mel spectrogram을 다음에 불러오기 쉽게 .pt 파일로 백업해놓은 것)
        self.use_cache = False  # 오디오를 mel spectrogram으로 변환하는 작업을 건너뛰고 캐시를 불러올지 설정
        self.save_cache = True  # 캐시 저장 여부 (오디오 파일명을 해시로 변환한 것이므로, 새로 저장 시 덮어쓰기가 됨)

# 파인튜닝 파라미터 설정
class FinetuneConfig:
    def __init__(self):
        # 음성 전처리 파라미터 (SSLConfig와 똑같이 맞춰주세요.)
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
        self.freeze_encoder = True

        # 훈련 파라미터
        self.batch_size = 64
        self.num_workers = 4
        self.epochs = 5   # default: 100
        self.learning_rate = 0.03
        self.weight_decay = 0.01

        # 캐시 사용 여부 (캐시 = ICBHI train data의 각 cycle의 mel spectrogram을 다음에 불러오기 쉽게 .pt 파일로 백업해놓은 것)
        self.use_cache = False  # 오디오를 mel spectrogram으로 변환하는 작업을 건너뛰고 캐시를 불러올지 설정
        self.save_cache = True  # 캐시 저장 여부 (오디오 파일명을 해시로 변환한 것이므로, 새로 저장 시 덮어쓰기가 됨)

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

        # 성능 평가 시 캐시 사용 여부 (캐시 = ICHBI test data의 각 cycle의 mel spectrogram)
        self.use_cache = False
        self.save_cache = True
        
        # 사전훈련 실험 이름
        self.step1_experiment_name = (
            f"{self.exp_name}-prt-"
            f"{self.ssl.batch_size}bs-{self.ssl.target_sr//1000}kHz-"
            f"top{self.ssl.top_k}-{self.ssl.K}K-"
            f"{self.ssl.dim_mlp}dim-{self.ssl.lambda_bce}ld-"
            f"{get_timestamp()}"
        )

        # 파인튜닝 실험 이름
        self.step2_experiment_name = (
            f"{self.exp_name}-fnt-"
            f"{self.finetune.batch_size}bs-{self.finetune.target_sr//1000}kHz-"
            f"{self.finetune.num_layers}ly-{self.finetune.dropout_rate}dr-"
            f"{get_timestamp()}"
        )

        # 성능평가 실험 이름
        self.step3_experiment_name = (
            f"{self.exp_name}-test-"
            f"{self.finetune.batch_size}bs-{self.finetune.target_sr//1000}kHz-"
            f"{self.finetune.num_layers}ly-{self.finetune.dropout_rate}dr-"
            f"{get_timestamp()}"
        )

        # train data를 사전훈련 set, 파인튜닝 set으로 분할할 때, 사전훈련 set의 비율
        self.split_ratio = 0.75

        # 파인튜닝 set을 다시 train set, valid set으로 분할할 때, valid set의 비율
        self.allow_val = False
        self.val_ratio = 0.25

        # wandb 설정
        self.wandb_project = "ICBHI_MLS_MoCo"
        self.wandb_entity = "boaz_woony-boaz"  # 이 줄은 변경하지 마세요.

        # 저장 경로
        self.checkpoint_dir = "checkpoints/exp001"
        self.log_dir = "logs/exp001"