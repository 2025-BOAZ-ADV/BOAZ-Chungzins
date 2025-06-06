"""실험 003: 강화된 데이터 증강 + 작은 배치 사이즈"""

from config.base_config import BaseConfig

class SSLConfig(BaseConfig):
    """실험 003의 특화된 설정
    BaseConfig에서 정의된 기본값을 상속받고, 
    실험에 특화된 값만 재정의합니다.
    """
    
    # Data preprocessing
    target_sr = 4000
    frame_size = 1024
    hop_length = 512
    n_mels = 128
    target_sec = 8
    
    # Data augmentation - 이 실험에만 특화된 설정
    augmentations = [
        {
            'type': 'SpecAugment',
            'params': {
                'time_mask_param': 0.9,  # 더 강한 time masking
                'freq_mask_param': 0.9   # 더 강한 frequency masking
            }
        },
        {
            'type': 'RandomCrop',
            'params': {
                'crop_size': 96  # 더 작은 crop size
            }
        },
        {
            'type': 'RandomNoise',
            'params': {
                'noise_level': 0.01  # 더 강한 노이즈
            }
        },
        {
            'type': 'PitchShift',
            'params': {
                'n_steps': 2  # 피치 시프트 추가
            }
        }
    ]
    
    # 기본값과 다른 학습 파라미터만 재정의
    batch_size = 64  # 더 작은 배치 사이즈
    epochs = 400    # 더 많은 에폭
    learning_rate = 0.01
    warmup_epochs = 15  # 더 긴 웜업

    # MoCo parameters
    K = 512
    m = 0.999
    T = 0.07
    dim_mlp = 128
    top_k = 10
    lambda_bce = 0.5

    # Wandb 설정
    wandb_project = "ADV-SSL"
