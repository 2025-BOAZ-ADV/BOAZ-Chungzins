import torch
import torch.nn as nn

from models.moco import MoCo
from models.backbone import create_backbone

class LungSoundClassifier(nn.Module):
    def __init__(self, encoder, classifier, freeze_encoder=True):
        """폐 소리 분류기
        
        Args:
            encoder: 사전훈련된 인코더
            classifier: 인코더에 이어붙일 분류기
            freeze_encoder: 인코더 가중치 고정 여부

        Returns:
            분류기가 결합된 백본 모델
        """
        super().__init__()
        
        self.encoder = encoder
        self.classifier = classifier
        self.freeze_encoder = freeze_encoder
        
        # 인코더의 가중기 고정 여부
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 백본에서 feature 추출
        features = self.encoder(x)
        # (Crackle, Wheeze) 분류
        return self.classifier(features)

def create_classifier(checkpoint_path, backbone_config, classifier, freeze_encoder=True):
    """분류기 생성
    
    Args:
        checkpoint_path: 사전학습된 가중치 경로
        backbone_config: MoCo 설정값
        classifier: 백본에 이어붙일 분류기
        freeze_encoder: 가중치 고정 여부
    
    Returns:
        LungSoundClassifier 인스턴스
    """
    # MoCo 백본 모델 생성
    backbone = MoCo(base_encoder=create_backbone, config=backbone_config)
    
    # 사전훈련된 가중치 로드
    checkpoint = torch.load(checkpoint_path)
    backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # MoCo에서 encoder_q만 가져와 분류 모델의 인코더로 사용
    encoder = backbone.encoder_q
    
    return LungSoundClassifier(encoder, classifier, freeze_encoder)
