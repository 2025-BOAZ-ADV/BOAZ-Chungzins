from src.models.SSL.mls import MLS
from src.models.backbone.resnet_audio import ResNet18
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/mls.yaml")

model = MLS(base_encoder_fn=lambda: ResNet18(), cfg=cfg.model)

# out = model(im_q, im_k, epochs)
# loss = out["loss"]