import torch.nn as nn
import torchvision.models as models

# ResNet18 Encoder
class ResNet18(nn.Module):
    def __init__(self, base="resnet18", pretrained=False):
        super().__init__()
        self.backbone = getattr(models, base)(pretrained=pretrained)
        # 첫 conv 채널 조정 등 필요 시 수정
    def forward(self, x):
        return self.backbone(x)

    
# ResNet50 Encoder		
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        # 1채널 입력으로 변경 (원래 3채널)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])  # (B, 2048, 1, 1)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (B, 2048)
        return x