from typing import Literal
from utils.logger import get_timestamp

from testing.config.data import AugmentationConfig, DataSplitConfig
from testing.config.models import ResnetConfig, TCNConfig, MLSMocoConfig, MocoConfig, ClassifierConfig
from testing.config.trainers import TrainerConfig

class CombinedConfig(AugmentationConfig, DataSplitConfig, ClassifierConfig, TrainerConfig):
    def __init__(
        self,
        backbone: Literal['resnet', 'tcn'],
        method: Literal['mls', 'moco']
    ):
        # 파라미터 상속
        super().__init__()

        # ------------- 빠른 실험을 위해 epoch만 변경 -------------
        self.pretrain_epochs = 4
        self.warmup_epochs = 1
        # ------------------------------------------------------
        
        # 실험 시드
        self.seed = 42

        # 백본 모델을 선택하여 해당 클래스의 파라미터 로드
        if backbone == "resnet":
            self._merge_config(ResnetConfig())
        elif backbone == "tcn":
            self._merge_config(TCNConfig())
        
        # SSL 방법을 선택하여 해당 클래스의 파라미터 로드
        if method == "mls":
            self._merge_config(MLSMocoConfig())
        elif method == 'moco':
            self._merge_config(MocoConfig())

        # 사전훈련 실험 이름
        self.step1_experiment_name = (
            f"prt-{self.batch_size}bs-{self.target_sr//1000}kHz-"
            f"top{self.top_k}-{self.K}K-"
            f"{self.dim_mlp}dim-{self.lambda_bce}ld-"
            f"{get_timestamp()}"
        )

        # 파인튜닝 실험 이름
        self.step2_experiment_name = (
            f"fnt-{self.batch_size}bs-{self.target_sr//1000}kHz-"
            f"{self.num_layers}ly-{self.dropout_rate}dr-"
            f"{get_timestamp()}"
        )

        # 성능평가 실험 이름
        self.step3_experiment_name = (
            f"test-{self.batch_size}bs-{self.target_sr//1000}kHz-"
            f"{self.num_layers}ly-{self.dropout_rate}dr-"
            f"{get_timestamp()}"
        )

        # wandb 설정
        self.wandb_project = "ICBHI_MLS_MoCo"
        self.wandb_entity = "boaz_woony-boaz"  # 이 줄은 변경하지 마세요.

        # 저장 경로
        self.checkpoint_dir = f"checkpoints"

    def _merge_config(self, backbone_cfg):
        for k, v in vars(backbone_cfg).items():
            setattr(self, k, v)