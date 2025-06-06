"""실험 002: 다양한 데이터 증강"""

from config.base_config import BaseConfig

class SSLConfig(BaseConfig):
    """실험 002의 특화된 설정
    다양한 데이터 증강을 사용하는 실험
    """
    
    # Data preprocessing
    target_sr = 4000
    frame_size = 1024
    hop_length = 512
    n_mels = 128
    target_sec = 8
    
    # Data augmentation - 다양한 증강 기법 사용
    augmentations = [
        {
            'type': 'SpecAugment',
            'params': {
                'time_mask_param': 0.8,
                'freq_mask_param': 0.8
            }
        },
        {
            'type': 'RandomCrop',
            'params': {
                'crop_size': 128
            }
        },
        {
            'type': 'RandomNoise',
            'params': {
                'noise_level': 0.005
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
    num_workers = 4
    epochs = 300
    learning_rate = 0.03
    weight_decay = 0.01
    warmup_epochs = 10

class FinetuneConfig:
    # Training parameters
    batch_size = 32
    num_workers = 4
    epochs = 100
    learning_rate = 0.001
    weight_decay = 0.0001
    
    # Model parameters
    freeze_backbone = True
    dropout_rate = 0.5
    
    # Data split
    specific_patient = 130  # finetune에 반드시 포함될 환자 ID
    use_weighted_sampler = True  # 클래스 불균형 처리
    
    # Data augmentation
    use_augmentation = True
    augmentations = [
        {
            'type': 'SpecAugment',
            'params': {
                'time_mask_param': 0.4,
                'freq_mask_param': 0.4
            }
        }
    ]

class ExperimentConfig:
    name = "exp002_augmentations"
    description = "다양한 데이터 증강 실험"
    
    # 데이터 분할
    ssl_ratio = 0.8
    val_ratio = 0.2
    
    # 설정
    ssl = SSLConfig
    finetune = FinetuneConfig
    
    # Wandb 설정
    wandb_project = "lung-sound-ssl"
    wandb_entity = None
    
    # 저장 경로
    checkpoint_dir = "checkpoints/exp002"
    log_dir = "logs/exp002"
