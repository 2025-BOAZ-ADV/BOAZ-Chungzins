from typing import Literal

from test_config.data import AugmentationConfig, DataSplitConfig
from test_config.models import ResnetConfig, TCNConfig, MLSMocoConfig, MocoConfig, ClassifierConfig
from test_config.trainers import TrainerConfig

class CombinedConfig(AugmentationConfig, DataSplitConfig, ClassifierConfig, TrainerConfig):
    def __init__(
        self,
        backbone: Literal['resnet', 'tcn'],
        method: Literal['mls', 'moco']
    ):
        # 파라미터 상속
        super().__init__()

        # ------------- 빠른 실험을 위해 epoch만 변경 -------------
        self.epochs = 4
        self.warmup_epochs = 1
        # ------------------------------------------------------
        
        # 실험 시드
        self.seed = 42

        # 백본 모델을 선택하여 해당 클래스의 파라미터 로드
        self.backbone = backbone
        if backbone == "resnet":
            self._merge_config(ResnetConfig())
        elif backbone == "tcn":
            self._merge_config(TCNConfig())
        
        # SSL 방법을 선택하여 해당 클래스의 파라미터 로드
        self.method = method
        if method == "mls":
            self._merge_config(MLSMocoConfig())
        elif method == 'moco':
            self._merge_config(MocoConfig())

        # wandb 설정
        self.wandb_project = "ICBHI_MLS_MoCo"
        self.wandb_entity = "boaz_woony-boaz"  # 이 줄은 변경하지 마세요.

        # 저장 경로
        self.checkpoint_dir = f"checkpoints"

    def _merge_config(self, backbone_cfg):
        for k, v in vars(backbone_cfg).items():
            setattr(self, k, v)