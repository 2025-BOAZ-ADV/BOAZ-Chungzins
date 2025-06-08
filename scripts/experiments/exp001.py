"""실험 001: 기본 SSL + Fine-tuning"""

from config.base_config import BaseConfig

class SSLConfig(BaseConfig):
    """실험 001의 특화된 설정
    기본적인 데이터 증강만 사용하는 베이스라인 실험
    """
    # Data preprocessing
    target_sr = 4000
    frame_size = 1024
    hop_length = 512
    n_mels = 128
    target_sec = 8
    # Data augmentation - 기본적인 SpecAugment만 사용
    augmentations = [
        {
            'type': 'SpecAugment',
            'params': {
                'time_mask_param': 0.8,
                'freq_mask_param': 0.8
            }
        }
    ]
    # MoCo parameters
    K = 512
    m = 0.999
    T = 0.07
    dim_mlp = 128
    top_k = 10
    lambda_bce = 0.5
    # Training parameters
    batch_size = 128
    num_workers = 2
    epochs = 300
    learning_rate = 0.03
    weight_decay = 0.01
    warmup_epochs = 10

class FinetuneConfig:
    # Training parameters
    batch_size = 128
    num_workers = 2
    epochs = 100
    learning_rate = 0.03
    weight_decay = 0.01
    # Model parameters
    freeze_backbone = True
    dropout_rate = 0.5
    # Data augmentation
    use_augmentation = False
    time_mask_ratio = 0.4
    freq_mask_ratio = 0.4

class ExperimentConfig:
    name = "exp001_basic_ssl_finetune"
    description = "기본 SSL + Fine-tuning 실험"
    # 랜덤 시드
    seed = 42
    # 데이터 분할 (pretrain - finetune)
    split_ratio = 0.8
    # 데이터 분할 (finetune 내에서 train - val)
    allow_val = False     # 현재는 X
    ssl_ratio = 0.7
    val_ratio = 0.3
    # 설정
    ssl = SSLConfig
    finetune = FinetuneConfig
    # Wandb 설정
    wandb_project = "lung-sound-classification"
    wandb_entity = "boaz_woony-boaz"     # 본인 wandb 계정명으로 변경
    # 저장 경로
    checkpoint_dir = "checkpoints/exp001"
    log_dir = "logs/exp001"
