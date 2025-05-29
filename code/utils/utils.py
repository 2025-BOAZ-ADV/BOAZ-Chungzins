import torchaudio
import math
import torchaudio.transforms as T



# cycle 로부터 label 설정
def get_class(cr, wh):
    if cr == 1 and wh == 1:
        return 3
    elif cr == 0 and wh == 1:
        return 2
    elif cr == 1 and wh == 0:
        return 1
    elif cr == 0 and wh == 0:
        return 0
    else:
        return -1
    

#  Mel Spectrogram 생성 ( sr=16KHz, frame_size=1024, hop_length=512, n_mels=128 )
def generate_mel_spectrogram(waveform, sample_rate, frame_size, hop_length, n_mels):
    if hop_length is None:
        hop_length = frame_size // 2
    mel_spec_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=frame_size,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spectrogram = mel_spec_transform(waveform)
    mel_db = T.AmplitudeToDB()(mel_spectrogram)
    return mel_db


 # 사이클 데이터 Repeat
def repeat_or_truncate_segment(mel_segment, target_frames):
    current_frames = mel_segment.shape[-1]
    if current_frames >= target_frames:
        return mel_segment[:, :, :target_frames]
    else:
        repeat_ratio = math.ceil(target_frames / current_frames)
        mel_segment = mel_segment.repeat(1, 1, repeat_ratio)
        return mel_segment[:, :, :target_frames]

        

# 데이터 Spec Augmentation ( 0~80% Random Masking )
def apply_spec_augment(mel_segment,args):
    M = mel_segment.shape[-1]
    F = mel_segment.shape[-2]

    # torchaudio의 마스킹은 0부터 mask_param까지 균등분포에서 랜덤하게 길이를 선택
    time_masking = T.TimeMasking(time_mask_param=int(M * args.time_mask_param))
    freq_masking = T.FrequencyMasking(freq_mask_param=int(F * args.freq_mask_param) )

    aug1 = freq_masking(mel_segment.clone())
    aug2 = time_masking(mel_segment.clone())
    aug3 = freq_masking(time_masking(mel_segment.clone()))

    return aug1, aug2, aug3


# Waveform resample ( 4KHz ~ 16KHz)
def resample_waveform(waveform, orig_sr, target_sr):
    
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        )
        return resampler(waveform), target_sr
    return waveform, orig_sr

import torch.nn.functional as F
import random

def preprocess_waveform_segment(waveform, unit_length):
    """unit_length 기준으로 waveform을 repeat + padding 또는 crop하여 길이 정규화"""
    waveform = waveform.squeeze(0)  # (1, L) → (L,) 로 바꿔도 무방
    length_adj = unit_length - len(waveform)

    if length_adj > 0:
        # waveform이 너무 짧은 경우 → repeat + zero-padding
        half_unit = unit_length // 2

        if length_adj < half_unit:
            # 길이 차이가 작으면 단순 padding
            half_adj = length_adj // 2
            waveform = F.pad(waveform, (half_adj, length_adj - half_adj))
        else:
            # 반복 후 부족한 부분 padding
            repeat_factor = unit_length // len(waveform)
            waveform = waveform.repeat(repeat_factor)[:unit_length]
            remaining = unit_length - len(waveform)
            half_pad = remaining // 2
            waveform = F.pad(waveform, (half_pad, remaining - half_pad))
    else:
        # waveform이 너무 길면 앞쪽 1/4 내에서 랜덤 crop
        length_adj = len(waveform) - unit_length
        start = random.randint(0, length_adj // 4)
        waveform = waveform[start:start + unit_length]

    return waveform.unsqueeze(0)  # 다시 (1, L)로


def test():
    print("test 1243")
