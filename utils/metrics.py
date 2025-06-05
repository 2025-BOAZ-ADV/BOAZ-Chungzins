import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, Tuple

def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """다중 레이블 분류 메트릭 계산
    
    Args:
        outputs: 모델 출력 (B, 2)
        labels: 실제 레이블 (B, 2)
    
    Returns:
        메트릭 딕셔너리
    """
    # 예측값 변환
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    
    # NumPy로 변환
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 메트릭 계산
    metrics = {
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_micro': f1_score(labels, predictions, average='micro'),
        'precision_macro': precision_score(labels, predictions, average='macro'),
        'precision_micro': precision_score(labels, predictions, average='micro'),
        'recall_macro': recall_score(labels, predictions, average='macro'),
        'recall_micro': recall_score(labels, predictions, average='micro')
    }
    
    return metrics

def get_confusion_matrices(outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, np.ndarray]:
    """각 레이블에 대한 혼동 행렬 계산
    
    Args:
        outputs: 모델 출력 (B, 2)
        labels: 실제 레이블 (B, 2)
    
    Returns:
        각 레이블의 혼동 행렬
    """
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    confusion_matrices = {
        'crackle': confusion_matrix(labels[:, 0], predictions[:, 0]),
        'wheeze': confusion_matrix(labels[:, 1], predictions[:, 1])
    }
    
    return confusion_matrices

def convert_to_single_label(multi_labels: torch.Tensor) -> torch.Tensor:
    """다중 레이블을 단일 레이블로 변환
    [0,0] -> 0 (Normal)
    [1,0] -> 1 (Crackle)
    [0,1] -> 2 (Wheeze)
    [1,1] -> 3 (Both)
    
    Args:
        multi_labels: (B, 2) 형태의 다중 레이블
    
    Returns:
        (B,) 형태의 단일 레이블
    """
    return (multi_labels[:, 0] * 1 + multi_labels[:, 1] * 2).long()

def calculate_per_class_metrics(outputs: torch.Tensor, 
                              labels: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """각 클래스별 메트릭 계산
    
    Args:
        outputs: 모델 출력 (B, 2)
        labels: 실제 레이블 (B, 2)
    
    Returns:
        클래스별 메트릭 딕셔너리
    """
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    single_labels = convert_to_single_label(torch.tensor(labels))
    single_preds = convert_to_single_label(torch.tensor(predictions))
    
    per_class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        mask = (single_labels == i)
        if mask.sum() > 0:
            per_class_metrics[class_name] = {
                'precision': precision_score(single_labels == i, single_preds == i),
                'recall': recall_score(single_labels == i, single_preds == i),
                'f1': f1_score(single_labels == i, single_preds == i)
            }
    
    return per_class_metrics
