import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Union
from torch.utils.data import Dataset
from collections import Counter
from pathlib import Path

def analyze_patient_distribution(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """환자별 호흡 사이클 상위 10명 분포 시각화
    
    Args:
        df: 파일명이 있는 DataFrame
    
    Returns:
        patient_stats: 환자별 통계
        fig: 분포 시각화
    """
    # 환자 ID 추출
    df['patient_id'] = df['filename'].apply(lambda x: x.split('_')[0])
    
    # 환자별 카운트
    patient_counts = df['patient_id'].value_counts()[:10]
    
    # 통계 계산
    patient_stats = pd.DataFrame({
        'patient_id': patient_counts.index,
        'cycle_count': patient_counts.values
    }).sort_values('cycle_count', ascending=False)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=patient_stats, x='patient_id', y='cycle_count', ax=ax)
    ax.set_title('Distribution of Respiratory Cycles per Patient')
    ax.set_xlabel('Patient ID')
    ax.set_ylabel('Number of Cycles')
    plt.xticks(rotation=45)
    
    return patient_stats, fig

def analyze_cycle_duration(dataset: Dataset) -> Tuple[Dict, plt.Figure]:
    """호흡 사이클 길이 분포 시각화
    
    Args:
        dataset: CycleDataset
    
    Returns:
        duration_stats: 길이 관련 통계
        fig: 분포 시각화
    """    
    # 길이 추출
    durations = [meta['duration'] for _, _, meta in dataset]
    
    # 통계 계산
    duration_stats = {
        'mean': np.mean(durations),
        'std': np.std(durations),
        'min': np.min(durations),
        'max': np.max(durations)
    }
    
    # 히스토그램 그리기
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(durations, bins=np.arange(0, max(durations)+1, 1))
    ax.set_title('Distribution of Respiratory Cycle Durations')
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Count')
    plt.xticks(np.arange(0, max(durations)+1, 1))
    
    return duration_stats, fig

def analyze_class_distribution(dataset: Dataset) -> Tuple[Dict[str, int], plt.Figure]:
    """클래스 분포 분석

    Args:
        dataset: CycleDataset 인스턴스

    Returns:
        class_counts: 클래스별 개수
        fig: matplotlib Figure 객체
    """
    labels = []
    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']

    for _, multi_label, _ in dataset:
        label_idx = int(multi_label[0]*1 + multi_label[1]*2)
        labels.append(class_names[label_idx])

    class_counts = Counter(labels)

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), ax=ax)
    ax.set_title('Distribution of Lung Sound Classes')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)

    return class_counts, fig

def visualize_mel_spectrograms(dataset: Dataset, num_samples: int = 5, save_dir: Union[str,Path] = None) -> None:
    """멜 스펙트로그램 시각화
    
    Args:
        dataset: CycleDataset 인스턴스
        num_samples: 시각화할 샘플 수
        save_dir: 이미지 저장 디렉토리
    """
    # 결과 저장 디렉토리
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    samples_per_class = {name: 0 for name in class_names}
    
    for idx in range(len(dataset)):
        mel, multi_label, meta = dataset[idx]
        label_idx = int(multi_label[0]*1 + multi_label[1]*2)
        label = class_names[label_idx]
        
        if samples_per_class[label] >= num_samples:
            continue
            
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(mel.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='magma')
        ax.set_title(f"{label} - {meta['filename']}")
        ax.set_xlabel('Time Frame')
        ax.set_ylabel('Mel Frequency Bin')
        plt.colorbar(im, ax=ax, format='%+2.0f dB')
        
        # Save figure
        plt.savefig(str(save_dir / f"{label}_{samples_per_class[label]}.png"))
        plt.close(fig)
        
        samples_per_class[label] += 1
        
        if all(count >= num_samples for count in samples_per_class.values()):
            break