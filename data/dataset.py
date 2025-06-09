import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal, Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from data.cache import DataCache
from data.preprocessor import resample_waveform, generate_mel_spectrogram, preprocess_waveform_segment, get_class
from config.config import Config


class CycleDataset(Dataset):
    """호흡 사이클들을 Mel Spectrogram으로 변환하여 저장
    Args:
        (작성 필요)
    Returns:
        mel (torch.Tensor): Mel Spectrogram (dB 스케일), augmentation 됐을 경우 (확인 필요)
        mel_data (Dict): 호흡 사이클의 메타데이터 (파일명, 호흡음 길이, 환자 번호, multi_label)
    """
    def __init__(
        self,
        data_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]],
        option: Literal["train", "test"],
        target_sr: int = 4000,
        target_sec: int = 8,
        frame_size: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        use_cache: bool = True,
        save_cache: bool = False
    ) -> None:
        self.data_path = Path(data_path)
        self.option = str(option)
        self.target_sr = target_sr
        self.target_sec = target_sec
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_cache = use_cache
        self.save_cache = save_cache
        
        cache_dir = Path("data/processed")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = DataCache(str(cache_dir))
        
        # train/test set 중 하나에서 모든 wav 파일명 가져오기
        metadata_path = Path(metadata_path)
        split_df = pd.read_csv(
            metadata_path / "train_test_split.txt",
            sep='\t',
            header=None,
            names=['filename', 'set']
        )
        self.file_list = split_df[split_df['set'] == self.option]['filename'].tolist()

        # 호흡 사이클 리스트 생성
        self.cycle_list = []

        print("[INFO] Preprocessing cycles...")
        for filename in tqdm(self.file_list):
            txt_path = self.data_path / f"{filename}.txt"
            wav_path = self.data_path / f"{filename}.wav"

            if not txt_path.exists() or not wav_path.exists():
                print(f"[WARNING] Missing file: {txt_path} or {wav_path}")
                continue

            # 주석 데이터 로드
            cycle_data = np.loadtxt(txt_path, usecols=(0, 1))
            lung_label = np.loadtxt(txt_path, usecols=(2, 3))

            # 청진음 데이터 로드
            waveform, orig_sr = torchaudio.load(wav_path)
            if waveform.shape[0] > 1:  # 스테레오를 모노로 변환
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 리샘플링
            waveform, _ = resample_waveform(waveform, orig_sr, self.target_sr)

            for idx in range(len(cycle_data)):
                # 호흡 주기 start, end
                start_sample = int(cycle_data[idx, 0] * self.target_sr)
                end_sample = int(cycle_data[idx, 1] * self.target_sr)
                lung_duration = cycle_data[idx, 1] - cycle_data[idx, 0]

                if end_sample <= start_sample:
                    continue
                
                # 캐시 키 생성
                cache_key = f"{filename}_{idx}"
                
                # 캐시된 mel spectrogram이 있으면 로드
                if self.use_cache and self.cache.exists(cache_key):
                    mel = self.cache.load(cache_key)

                else:
                    # 호흡 사이클 분할
                    cycle_wave = waveform[:, start_sample:end_sample]

                    # 호흡 사이클 길이 고정
                    normed_wave = preprocess_waveform_segment(cycle_wave, unit_length=int(self.target_sec * self.target_sr))

                    # Mel Spectrogram으로 변환
                    mel = generate_mel_spectrogram(normed_wave, sample_rate=self.target_sr, frame_size=self.frame_size, hop_length=self.hop_length, n_mels=self.n_mels)
                    
                    # 캐시에 저장
                    if self.save_cache:
                        self.cache.save(cache_key, mel)

                # 라벨 생성
                cr = int(lung_label[idx, 0])
                wh = int(lung_label[idx, 1])
                label = get_class(cr, wh)
                
                # multi-label로 변환
                multi_label = torch.tensor([
                    float(label in [1, 3]),
                    float(label in [2, 3])
                ])

                # 환자 ID 추출
                patient_id = int(filename.split('_')[0])

                # 메타데이터 생성
                meta_data = {
                    'filename': filename,
                    'duration': lung_duration,
                    'patient_id': patient_id,
                }
                
                self.cycle_list.append((mel, multi_label, meta_data))

        print(f"[INFO] Total cycles collected: {len(self.cycle_list)}")

    def __len__(self) -> int:
        return len(self.cycle_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        mel, multi_label, meta_data = self.cycle_list[idx]
            
        return mel, multi_label, meta_data


class MoCoCycleDataset(CycleDataset):
    """현재 사용하지 않습니다."""
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        mel, multi_label, meta_data = self.cycle_list[idx]
        
        if self.transform:
            mel_q = self.transform(mel.clone())
            mel_k = self.transform(mel.clone())
        else:
            mel_q = mel
            mel_k = mel
        
        return mel_q, mel_k, meta_data['patient_id']
