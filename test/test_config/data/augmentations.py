from test_config.data.preprocess import PreprocessConfig

class AugmentationConfig(PreprocessConfig):
    def __init__(
        self,
        augmentations = None
    ):
        super().__init__()

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
                        'crop_size': int(self.target_sr * self.target_sec)
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