import torch
import torch.nn as nn
from models.backbone import create_backbone

class LungSoundClassifier(nn.Module):
    def __init__(self, backbone, num_classes=2, freeze_backbone=True, dropout_rate=0.5):
        """폐 소리 분류기
        
        Args:
            backbone: 사전학습된 백본 모델
            num_classes: 출력 클래스 수 (crackle, wheeze)
            freeze_backbone: 가중치 고정 여부
            dropout_rate: Dropout 비율

        Returns:
            분류기가 결합된 백본 모델
        """
        super().__init__()
        self.backbone = backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 백본에서 feature 추출
        features = self.backbone(x)
        # 분류
        return self.classifier(features)

def create_classifier(checkpoint_path=None, num_classes=2, freeze_backbone=True, dropout_rate=0.5):
    """분류기 생성
    
    Args:
        checkpoint_path: 사전학습된 가중치 경로
        num_classes: 출력 클래스 수
        freeze_backbone: 가중치 고정 여부
        dropout_rate: Dropout 비율
    
    Returns:
        LungSoundClassifier 인스턴스
    """
    backbone = create_backbone()
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        # MoCo의 encoder_q에서 가중치 로드
        backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    return LungSoundClassifier(backbone, num_classes, freeze_backbone, dropout_rate)
