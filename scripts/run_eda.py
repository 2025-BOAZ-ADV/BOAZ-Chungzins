"""EDA 스크립트"""

import os
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from data.dataset import CycleDataset
from config.base_config import BaseConfig
from utils.eda import (
    analyze_patient_distribution,
    analyze_cycle_duration,
    analyze_class_distribution,
    visualize_mel_spectrograms
)
from utils.logger import get_timestamp

def save_statistics(stats: dict, filepath: str):
    """통계 결과를 텍스트 파일로 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for key, value in stats.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")

def main():    
    # 현재 프로젝트 루트 디렉토리 설정
    project_root = Path(__file__).parent.parent

    # 결과 저장 디렉토리 생성
    out_dir = project_root / 'eda_results' / str(get_timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 경로 설정
    data_path = project_root / 'data' / 'raw'
    metadata_path = project_root / 'data' / 'metadata'

    # 데이터셋 로드
    train_dataset = CycleDataset(
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        option="train",
        target_sr=BaseConfig.target_sr,
        target_sec=BaseConfig.target_sec,
        frame_size=BaseConfig.frame_size,
        hop_length=BaseConfig.hop_length,
        n_mels=BaseConfig.n_mels,
        use_cache=True,    
        save_cache=False
    )

    test_dataset = CycleDataset(
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        option="test",
        target_sr=BaseConfig.target_sr,
        target_sec=BaseConfig.target_sec,
        frame_size=BaseConfig.frame_size,
        hop_length=BaseConfig.hop_length,
        n_mels=BaseConfig.n_mels,
        use_cache=True,    
        save_cache=False
    )

    # metadata DataFrame 생성
    train_df = pd.DataFrame([meta for _, _, meta in train_dataset.cycle_list])
    test_df = pd.DataFrame([meta for _, _, meta in test_dataset.cycle_list])
    
    data_list = ['train', 'test']

    # 1. 환자별 분포 분석
    print("1. Analyzing patient distribution...")
    for idx, df in enumerate([train_df, test_df]):
        patient_stats, fig = analyze_patient_distribution(df)
        fig.savefig(str(out_dir / f"Top10_{data_list[idx]}_patient_distribution.png"))
        plt.close(fig)
        save_statistics(
            {'patient_distribution': patient_stats.to_dict()},
            str(out_dir / f"Top10_{data_list[idx]}_patient_stats.txt")
        )
    
    # 2. 호흡 사이클 길이 분석
    print("2. Analyzing cycle durations...")
    for idx, dataset in enumerate([train_dataset, test_dataset]):
        duration_stats, fig = analyze_cycle_duration(dataset)
        fig.savefig(str(out_dir / f"{data_list[idx]}_cycle_duration_distribution.png"))
        plt.close(fig)
        save_statistics(
            {'cycle_duration': duration_stats},
            str(out_dir / f"{data_list[idx]}_duration_stats.txt")
        )
    
    # 3. 클래스 분포 분석
    print("3. Analyzing class distribution...")
    for idx, dataset in enumerate([train_dataset, test_dataset]):
        class_counts, fig = analyze_class_distribution(dataset)
        fig.savefig(str(out_dir / f"{data_list[idx]}_class_distribution.png"))
        plt.close(fig)
        save_statistics(
            {'class_distribution': class_counts},
            str(out_dir / f"{data_list[idx]}_class_stats.txt")
        )
    
    # 4. 멜 스펙트로그램 시각화
    print("4. Visualizing mel spectrograms...")
    spec_dir = out_dir / "spectrograms"
    spec_dir.mkdir(exist_ok=True)
    visualize_mel_spectrograms(
        dataset,
        num_samples=5,
        save_dir=str(spec_dir)
    )
    
    print(f"EDA results saved to {out_dir}")

if __name__ == "__main__":
    main()