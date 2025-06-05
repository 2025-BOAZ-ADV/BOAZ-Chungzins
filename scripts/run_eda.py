"""데이터 탐색적 분석 스크립트"""

import os
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from data.dataset import CycleDataset
from utils.eda import (
    analyze_patient_distribution,
    analyze_cycle_duration,
    analyze_class_distribution,
    visualize_mel_spectrograms
)

def parse_args():
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='데이터 디렉토리 경로')
    parser.add_argument('--out-dir', type=str, default='eda_results',
                        help='결과 저장 디렉토리')
    return parser.parse_args()

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
    args = parse_args()
      # 결과 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 로드
    dataset = CycleDataset(data_path=args.data_dir)
    
    # metadata DataFrame 생성
    df = pd.DataFrame([meta for _, meta in dataset.cycle_list])
    
    # 1. 환자별 분포 분석
    print("Analyzing patient distribution...")
    patient_stats, fig = analyze_patient_distribution(df)
    fig.savefig(str(out_dir / "patient_distribution.png"))
    plt.close(fig)
    save_statistics(
        {'patient_distribution': patient_stats.to_dict()},
        str(out_dir / "patient_stats.txt")
    )
      # 2. 호흡 사이클 길이 분석
    print("Analyzing cycle durations...")
    duration_stats, fig = analyze_cycle_duration(dataset)
    fig.savefig(str(out_dir / "cycle_duration_distribution.png"))
    plt.close(fig)
    save_statistics(
        {'cycle_duration': duration_stats},
        str(out_dir / "duration_stats.txt")
    )
    
    # 3. 클래스 분포 분석
    print("Analyzing class distribution...")
    class_counts, fig = analyze_class_distribution(dataset)
    fig.savefig(str(out_dir / "class_distribution.png"))
    plt.close(fig)
    save_statistics(
        {'class_distribution': class_counts},
        str(out_dir / "class_stats.txt")
    )
    
    # 4. 멜 스펙트로그램 시각화
    print("Visualizing mel spectrograms...")
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
