import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url

def create_backbone():
    """ResNet50 백본 모델 생성
    Returns:
        nn.Module: 수정된 ResNet50 모델
    """
    # 1. 기본 ResNet50 생성 (pretrained=False로 시작)
    resnet = models.resnet50(pretrained=False)

    # 2. 첫 번째 conv 레이어를 1채널용으로 수정
    resnet.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # 먼저 fc 제거 (feature extractor로 사용)
    resnet.fc = nn.Identity()

    # 3. ImageNet 가중치 로드 (conv1 제외)
    state_dict = load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        progress=True
    )
    if 'conv1.weight' in state_dict:
        del state_dict['conv1.weight']
    resnet.load_state_dict(state_dict, strict=False)

    return resnet
