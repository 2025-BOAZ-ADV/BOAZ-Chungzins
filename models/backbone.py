import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url

def create_backbone():
    """ResNet50 백본 모델 생성
    Returns:
        nn.Module: 수정된 ResNet50 모델
    """
    # 1. 기본 ResNet50 생성 (weights=None)
    try:
        resnet = models.resnet50(weights=None)
    except TypeError:
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

    # 3. fc를 임시로 Linear로 맞춰놓음 (1000개 클래스)
    resnet.fc = nn.Linear(2048, 1000)

    # 4. ImageNet 가중치 로드 (conv1, fc 제외)
    state_dict = load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        progress=True
    )
    for key in ['conv1.weight', 'fc.weight', 'fc.bias']:
        if key in state_dict:
            del state_dict[key]
    resnet.load_state_dict(state_dict, strict=False)

    # 5. feature extractor로 사용
    resnet.fc = nn.Linear(2048, 2048)

    return resnet
