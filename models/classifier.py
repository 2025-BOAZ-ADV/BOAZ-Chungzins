import torch
import torch.nn as nn
from models.backbone import create_backbone

class LungSoundClassifier(nn.Module):
    def __init__(self, backbone, classifier, freeze_backbone=True):
        """폐 소리 분류기
        
        Args:
            backbone: 사전학습된 백본 모델
            classifier: 백본에 이어붙일 분류기
            freeze_backbone: 가중치 고정 여부

        Returns:
            분류기가 결합된 백본 모델
        """
        super().__init__()
        
        self.backbone = backbone
        self.classifier = classifier
        self.freeze_backbone = freeze_backbone
        
        # 가중기 고정 여부
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 백본에서 feature 추출
        features = self.backbone(x)
        # (Crackle, Wheeze) 분류
        return self.classifier(features)

def create_classifier(checkpoint_path=None, classifier=None, freeze_backbone=True):
    """분류기 생성
    
    Args:
        checkpoint_path: 사전학습된 가중치 경로
        classifier: 백본에 이어붙일 분류기
        freeze_backbone: 가중치 고정 여부
    
    Returns:
        LungSoundClassifier 인스턴스
    """
    backbone = create_backbone()
    
    # MoCo 가중치 로드
    checkpoint = torch.load(checkpoint_path)
    backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # MoCo에서 encoder_q만 가져와 backbone으로 사용
    backbone = backbone.encoder_q.eval()
    
    return LungSoundClassifier(backbone, classifier, freeze_backbone)
