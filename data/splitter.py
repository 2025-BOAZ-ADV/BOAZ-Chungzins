import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Union, Literal

import torch
from torch.utils.data import Subset, Dataset, DataLoader

def get_shuffled_filenames(metadata_path: Union[str, Path], option: Literal["pretrain", "finetune"],
                            split_ratio: float = 0.8, seed: int = 42) -> List[str]:
    """ICBHI train data를 랜덤 셔플하여 지정된 비율만큼 파일명 리스트 반환

    Args:
        metadata_path: train_test_split.txt가 있는 경로
        option: pretrain or finetune
        split_ratio: train data를 분할하는 비율
        seed: random seed

    Returns:
        Subset: filename list
    """
    random.seed(seed)

    # load train data
    metadata_path = Path(metadata_path)
    split_df = pd.read_csv(
        metadata_path / "train_test_split.txt",
        sep='\t',
        header=None,
        names=['filename', 'set']
    )
    train_df = split_df[split_df['set'] == 'train']

    # shuffle train data
    shuffled_df = train_df.sample(frac=1, random_state=seed)

    # split ratio
    train_size = int(split_ratio * len(shuffled_df))

    if option == "pretrain":
        splitted_df = shuffled_df[:train_size]
    elif option == "finetune":
        splitted_df = shuffled_df[train_size:]

    # filename list
    filename_list = splitted_df['filename'].tolist()

    return filename_list

def split_cycledataset(dataset: Dataset, filename_list: List[str], seed: int = 42) -> Subset:
    """CycleDataset을 pretrain용 또는 finetuning용으로 분할
    
    Args:
        dataset: CycleDataset
        filename_list: 파일명 리스트
    
    Returns:
        shuffled_cycle_subset: 분할 및 셔플된 CycleDataset
    """
    random.seed(seed)
    np.random.seed(seed)

    file_idx = []
    for i in range(len(dataset)):
        filename = dataset[i][2]['filename']
        file_idx.append(i)
    
    random.shuffle(file_idx)
    shuffled_cycle_subset = Subset(dataset, file_idx)

    return shuffled_cycle_subset

def create_dataloaders(pretext_dataset: Dataset, 
                      finetune_dataset: Dataset,
                      batch_size: int,
                      num_workers: int) -> Dict[str, DataLoader]:
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
        'train': DataLoader(
            pretext_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        ),
        'val': DataLoader(
            finetune_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
    }
    
    return dataloaders


##### 아래 함수들은 사용 여부 불확실
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
