import random
import pandas as pd
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Subset, Dataset, DataLoader
from config.config import Config

def split_by_patient(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """환자 ID 기준으로 데이터 분할
    
    Args:
        df: 파일명과 환자 ID가 있는 DataFrame
        test_size: 테스트 세트 비율
        seed: 랜덤 시드
    
    Returns:
        train_files, test_files: 분할된 파일명 리스트
    """
    random.seed(seed)
    
    # 환자 번호 추출
    patient_numbers = []
    for filename in df['filename']:
        number = int(filename.split('_')[0])
        patient_numbers.append(number)
    
    # 환자별 파일 수 계산
    patient_counts = pd.Series(patient_numbers).value_counts()
    all_patient_ids = list(patient_counts.index)
    
    # 환자 ID로 분할
    random.shuffle(all_patient_ids)
    split_idx = int(len(all_patient_ids) * (1 - test_size))
    train_patients = all_patient_ids[:split_idx]
    test_patients = all_patient_ids[split_idx:]
    
    # 파일명 리스트 생성
    train_files = df[df['filename'].apply(lambda x: int(x.split('_')[0]) in train_patients)]['filename'].tolist()
    test_files = df[df['filename'].apply(lambda x: int(x.split('_')[0]) in test_patients)]['filename'].tolist()
    
    return train_files, test_files

def split_ssl_finetune(dataset: Dataset, pretext_files: List[str], finetune_files: List[str]) -> Tuple[Subset, Subset]:
    """데이터셋을 SSL pretext와 fine-tuning용으로 분할
    
    Args:
        dataset: 전체 데이터셋
        pretext_files: pretext 학습용 파일명 리스트
        finetune_files: fine-tuning용 파일명 리스트
    
    Returns:
        pretext_dataset, finetune_dataset: 분할된 데이터셋
    """
    pretext_idx = []
    finetune_idx = []
    
    for i in range(len(dataset)):
        filename = dataset[i][2][0]  # meta data에서 파일명 추출
        if filename in pretext_files:
            pretext_idx.append(i)
        elif filename in finetune_files:
            finetune_idx.append(i)
    
    random.shuffle(pretext_idx)
    random.shuffle(finetune_idx)
    
    return Subset(dataset, pretext_idx), Subset(dataset, finetune_idx)

def create_dataloaders(pretext_dataset: Dataset, 
                      finetune_dataset: Dataset,
                      batch_size: int = Config.batch_size,
                      num_workers: int = Config.num_workers) -> Dict[str, DataLoader]:
    """데이터로더 생성
    
    Args:
        pretext_dataset: SSL pretext 학습용 데이터셋
        finetune_dataset: fine-tuning용 데이터셋
        batch_size: 배치 크기
        num_workers: 워커 수
    
    Returns:
        Dictionary containing DataLoaders
    """
    dataloaders = {
        'pretext': DataLoader(
            pretext_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        ),
        'finetune': DataLoader(
            finetune_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
    }
    
    return dataloaders

def get_class_weights(dataset: Dataset) -> torch.Tensor:
    """클래스별 가중치 계산
    
    Args:
        dataset: 데이터셋
    
    Returns:
        class_weights: 클래스별 가중치
    """
    labels = []
    for _, label, _ in dataset:
        if torch.equal(label, torch.tensor([0., 0.])):
            labels.append(0)  # Normal
        elif torch.equal(label, torch.tensor([1., 0.])):
            labels.append(1)  # Crackle
        elif torch.equal(label, torch.tensor([0., 1.])):
            labels.append(2)  # Wheeze
        elif torch.equal(label, torch.tensor([1., 1.])):
            labels.append(3)  # Both

    label_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / label_counts.float()
    return class_weights
