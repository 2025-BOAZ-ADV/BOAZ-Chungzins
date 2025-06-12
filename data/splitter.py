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

    print(f"[INFO] {option} dataset has {len(filename_list)} audio files & ", end="")
    return filename_list

def split_cycledataset(train_dataset: Dataset, filename_list: List[str], seed: int = 42) -> Subset:
    """CycleDataset을 pretrain용 또는 finetuning용으로 분할
    
    Args:
        train_dataset: CycleDataset
        filename_list: 파일명 리스트
    
    Returns:
        shuffled_cycle_subset: 분할 및 셔플된 CycleDataset
    """
    random.seed(seed)
    np.random.seed(seed)

    cycle_idx = []
    for i in range(len(train_dataset)):
        filename = train_dataset[i][2]['filename']

        # filename_list에 있는 .wav 파일의 cycle들만 추가
        if filename in filename_list:
            cycle_idx.append(i)
    
    random.shuffle(cycle_idx)
    shuffled_cycle_subset = Subset(train_dataset, cycle_idx)

    print(f"{len(shuffled_cycle_subset)} cycles.")
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


### 아래 함수는 현재 사용하지 않음
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
