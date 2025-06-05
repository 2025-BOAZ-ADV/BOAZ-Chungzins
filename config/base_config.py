"""기본 설정 클래스"""

class BaseConfig:
    """모든 실험에서 공통으로 사용되는 기본 설정"""
    
    # Data preprocessing
    target_sr = 4000  # 4KHz
    frame_size = 1024
    hop_length = 512  # frame_size 절반
    n_mels = 128
    target_sec = 8

    # MoCo parameters
    K = 512  # queue size
    m = 0.999  # momentum of updating key encoder
    T = 0.07  # softmax temperature
    dim_mlp = 128
    top_k = 10
    lambda_bce = 0.5    # Default training parameters
    batch_size = 128
    num_workers = 4
    epochs = 300
    learning_rate = 0.03
    momentum = 0.9
    weight_decay = 0.01
    warmup_epochs = 10

    # Wandb default settings
    wandb_project = "ADV-SSL"

    @classmethod
    def get_config_dict(cls):
        """설정값들을 딕셔너리로 반환"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') and not callable(v)}
