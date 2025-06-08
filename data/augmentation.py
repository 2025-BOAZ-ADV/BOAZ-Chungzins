import torch
import torchaudio.transforms as T
import random
import math
from typing import List, Dict, Any, Optional
from config.config import Config

class AugmentationBase:
    """증강 기법의 기본 클래스"""
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SpecAugment(AugmentationBase):
    """SpecAugment 증강"""
    def __init__(self, time_mask_param: float = 0.8, freq_mask_param: float = 0.8):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        M = mel.shape[-1]  # 시간 축 길이
        F = mel.shape[-2]  # 주파수 축 길이
        
        time_masking = T.TimeMasking(time_mask_param=int(M * self.time_mask_param))
        freq_masking = T.FrequencyMasking(freq_mask_param=int(F * self.freq_mask_param))
        
        mel = freq_masking(mel.clone())
        mel = time_masking(mel)
        return mel

class RandomCrop(AugmentationBase):
    """Random Crop 증강"""
    def __init__(self, crop_size: int):
        self.crop_size = crop_size
        
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.shape[-1] <= self.crop_size:
            return mel
        
        start = torch.randint(0, mel.shape[-1] - self.crop_size + 1, (1,)).item()
        return mel[:, :, start:start + self.crop_size]

class RandomNoise(AugmentationBase):
    """가우시안 노이즈 추가"""
    def __init__(self, noise_level: float = 0.005):
        self.noise_level = noise_level
        
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(mel) * self.noise_level
        return mel + noise

class PitchShift(AugmentationBase):
    """피치 시프트"""
    def __init__(self, n_steps: int = 2):
        self.n_steps = n_steps
        
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-self.n_steps, self.n_steps)
        if shift > 0:
            mel = torch.cat([mel[:, shift:, :], mel[:, :shift, :]], dim=1)
        elif shift < 0:
            shift = abs(shift)
            mel = torch.cat([mel[:, -shift:, :], mel[:, :-shift, :]], dim=1)
        return mel

class TimeStretch(AugmentationBase):
    """시간 축 스트레칭"""
    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.2):
        self.min_rate = min_rate
        self.max_rate = max_rate
        
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        rate = random.uniform(self.min_rate, self.max_rate)
        if rate == 1.0:
            return mel
            
        # 시간 축 크기 조정
        orig_size = mel.shape[-1]
        target_size = int(orig_size * rate)
        mel_stretched = torch.nn.functional.interpolate(
            mel, size=(mel.shape[1], target_size),
            mode='bilinear', align_corners=False
        )
        
        # 원래 길이로 자르거나 패딩
        if target_size > orig_size:
            return mel_stretched[:, :, :orig_size]
        else:
            padding = orig_size - target_size
            return torch.nn.functional.pad(mel_stretched, (0, padding))

class AugmentationComposer:
    """여러 증강 기법을 조합"""
    def __init__(self, augmentations: List[Dict[str, Any]]):
        """
        Args:
            augmentations: 증강 설정 리스트
            예시: [
                {'type': 'SpecAugment', 'params': {'time_mask_param': 0.8}},
                {'type': 'RandomCrop', 'params': {'crop_size': 128}}
            ]
        """
        self.augmentations = []
        for aug in augmentations:
            aug_type = aug['type']
            aug_params = aug.get('params', {})
            
            if aug_type == 'SpecAugment':
                self.augmentations.append(SpecAugment(**aug_params))
            elif aug_type == 'RandomCrop':
                self.augmentations.append(RandomCrop(**aug_params))
            elif aug_type == 'RandomNoise':
                self.augmentations.append(RandomNoise(**aug_params))
            elif aug_type == 'PitchShift':
                self.augmentations.append(PitchShift(**aug_params))
    
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """증강 적용"""
        # 이미 정의된 증강들 중에서 랜덤하게 선택하여 적용
        aug_indices = random.sample(range(len(self.augmentations)), 
                                  k=random.randint(1, len(self.augmentations)))
        
        mel_aug = mel.clone()
        for idx in aug_indices:
            mel_aug = self.augmentations[idx](mel_aug)
        
        return mel_aug
    
    def generate_views(self, mel: torch.Tensor, num_views: int = 2) -> List[torch.Tensor]:
        """MoCo를 위한 multiple views 생성
        
        Args:
            mel (torch.Tensor): 입력 Mel spectrogram
            num_views (int): 생성할 views 수 (기본값: 2)
            
        Returns:
            List[torch.Tensor]: 증강된 views 리스트
        """
        return [self.__call__(mel) for _ in range(num_views)]

def create_augmenter(config: Config, augmentations: [List[Dict[str, Any]]]) -> AugmentationComposer:
    """증강기 생성
    
    Args:
        config: 실험 설정
        augmentations: 증강 설정 리스트, dictionary로 구성된 리스트 (없으면 기본값 사용)
        예시: [
                {'type': 'SpecAugment', 'params': {'time_mask_param': 0.8}},
                {'type': 'RandomCrop', 'params': {'crop_size': 128}}
            ]
        
    Returns:
        AugmentationComposer 인스턴스
    """
    if augmentations is None:
        augmentations = [
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
                    'crop_size': int(config.target_sr * config.target_sec)
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
    
    return AugmentationComposer(augmentations)

def apply_spec_augment(mel_segment: torch.Tensor):
    """
    Mel spectrogram에 SpecAugment (frequency, time masking) 적용
    여러 증강 버전을 반환
    """
    M = mel_segment.shape[-1]
    F = mel_segment.shape[-2]

    time_masking = T.TimeMasking(time_mask_param=int(M * 0.8))
    freq_masking = T.FrequencyMasking(freq_mask_param=int(F * 0.8))

    aug1 = freq_masking(time_masking(mel_segment.clone()))
    aug2 = freq_masking(time_masking(mel_segment.clone()))

    return aug1, aug2


### 사용하지 않는 함수
def repeat_or_truncate_segment(mel_segment: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Mel spectrogram 세그먼트를 target_frames에 맞게 조정"""
    current_frames = mel_segment.shape[-1]
    if current_frames >= target_frames:
        return mel_segment[:, :, :target_frames]
    else:
        repeat_ratio = math.ceil(target_frames / current_frames)
        mel_segment = mel_segment.repeat(1, 1, repeat_ratio)
        return mel_segment[:, :, :target_frames]