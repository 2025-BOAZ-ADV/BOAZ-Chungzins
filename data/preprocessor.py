import torch
import torchaudio
import torchaudio.transforms as T
from config.config import Config

def get_class(cr, wh):
    """폐 소리 클래스 반환
    Args:
        cr (int): crackle 여부 (0 or 1)
        wh (int): wheeze 여부 (0 or 1)
    Returns:
        int: 폐 소리 클래스 (0: normal, 1: crackle, 2: wheeze, 3: both)
    """
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

def generate_mel_spectrogram(waveform, sample_rate, frame_size, hop_length, n_mels):
    """Mel Spectrogram 생성
    Args:
        waveform (torch.Tensor): 오디오 파형
        sample_rate (int): 샘플링 레이트
        frame_size (int): FFT 크기 (=윈도우 길이)
        hop_length (int): 프레임 간 간격 (작을수록 더 촘촘하게 분석, 보통 frame_size의 절반)
        n_mels (int): mel filter 개수
    Returns:
        torch.Tensor: Mel Spectrogram (dB 스케일)
    """
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

def resample_waveform(waveform, orig_sr, target_sr):
    """오디오 파형 리샘플링
    Args:
        waveform (torch.Tensor): 원본 오디오 파형
        orig_sr (int): 원본 샘플링 레이트
        target_sr (int): 목표 샘플링 레이트
    Returns:
        tuple: (리샘플된 파형, 새로운 샘플링 레이트)
    """
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        )
        return resampler(waveform), target_sr
    return waveform, orig_sr
