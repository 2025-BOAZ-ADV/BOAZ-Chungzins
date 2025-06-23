from typing import List, Dict, Union, Any

class AugmentationConfig:
    def __init__(
        self,
        target_sr: int = 4000,
        target_sec: Union[int, float] = 8,
        augmentations: List[Dict[str, Any]] = None
    ):
        '''Augmentation 파라미터'''
        if augmentations is not None:
            self.augmentations = augmentations
        else:
            self.augmentations = [
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
                        'crop_size': int(target_sr * target_sec)
                    }
                },
                {
                    'type': 'RandomNoise',
                    'params': {
                        'noise_level': 0.005
                    }
                },
                {
                    'type': 'TimeStretch',
                    'params': {
                        'min_rate': 0.8,
                        'max_rate': 1.2
                    }
                }
            ]