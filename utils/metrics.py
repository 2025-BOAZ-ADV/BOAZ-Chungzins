import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, Tuple

def get_confusion_matrix_for_multi_label(all_labels, all_preds):
    """각 label에 대한 혼동 행렬 계산
    
    Args:
        all_labels: 정답 multi_label
        all_preds: 정답 multi_label
    
    Returns:
        각 단일 label의 혼동 행렬 (Dict)
    """
    
    confusion_matrices = {
        'crackle': confusion_matrix(all_labels[:, 0], all_preds[:, 0]),
        'wheeze': confusion_matrix(all_labels[:, 1], all_preds[:, 1])
    }
    
    return confusion_matrices

def log_confusion_matrix_for_multi_label(confusion_matrices, avg_results, logger=None):
    """각 label에 대한 혼동 행렬을 로그
    
    Args:
        confusion_matrices: 각 단일 label의 혼동 행렬 (Dict)
        avg_results: 각 label별 평균 성능 (Dict)
        logger: WandbLogger
    """
    label_names = ['Crackle', 'Wheeze']

    for label_name in label_names:
        cm = confusion_matrices[label_name]

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['0', '1'], yticklabels=['0', '1'], ax=ax)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix of {label_name}')

        plt.tight_layout()

        if logger:
            logger.log({f"confusion_matrix of {label_name}": wandb.Image(fig)})

            logger.log({
                "Test/avg_sens_2label": avg_results["sensitivity"],
                "Test/avg_spec_2label": avg_results["specificity"],
                "Test/avg_score_2label": avg_results["ICBHI score"]
            })

        plt.close(fig)

def convert_to_multi_class(labels):
    """다중 레이블 -> 다중 클래스 변환
    [0,0] -> 0 (Normal)
    [1,0] -> 1 (Crackle)
    [0,1] -> 2 (Wheeze)
    [1,1] -> 3 (Both)
    """
    
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    return (labels[:, 0] * 1 + labels[:, 1] * 2).astype(int)

def get_confusion_matrix_for_multi_class(all_labels, all_preds):
    """다중 레이블 -> 다중 클래스로 변환하여 혼동 행렬 계산
    
    Args:
        outputs: 모델 출력
        labels: 실제 레이블
    
    Returns:
        클래스별 메트릭 딕셔너리
    """
    
    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    all_labels_cls = convert_to_multi_class(all_labels)
    all_preds_cls = convert_to_multi_class(all_preds)

    # 4x4 matrix
    conf_matrix = confusion_matrix(all_labels_cls, all_preds_cls, labels=[0,1,2,3])

    # Positive: 1,2,3 / Negative: 0
    TP = conf_matrix[1:, 1:].sum()    # 양성 중에 양성으로 예측
    FN = conf_matrix[1:, 0].sum()     # 양성인데 음성으로 예측
    FP = conf_matrix[0, 1:].sum()     # 음성인데 양성으로 예측
    TN = conf_matrix[0, 0]            # 음성인데 양성으로 예측

    sens = TP / (TP + FN + 1e-6)    # 민감도
    spec = TN / (TN + FP + 1e-6)    # 특이도

    print("4x4 Confusion Matrix:\n", conf_matrix)
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"ICBHI Score: {(sens+spec)/2:.4f}")

    return conf_matrix, sens, spec

def log_confusion_matrix_for_multi_class(conf_matrix, sens, spec, logger=None):

    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Multi-Class 4x4 Confusion Matrix')

    plt.text(
            0.99, 0.16,  # 우하단 (x=99%, y=16%) 위치
            f"Sensitivity: {sens*100:.2f}\nSpecificity: {spec*100:.2f}\nICBHI Score: {100*(sens+spec)/2:.2f}",
            ha='right', va='bottom',
            transform=plt.gca().transAxes,  # 축 기준 좌표로 해석
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    plt.tight_layout()

    if logger:
        logger.log({"multiclass_confusion_matrix": wandb.Image(fig)})

        logger.log({
            "Metrics/sensitivity_4class": sens,
            "Metrics/specificity_4class": spec,
            "Metrics/ICHBI_score_4class": (sens+spec)/2
        })

    plt.close(fig)


##### 사용하지 않음 #####
def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """다중 레이블 분류 메트릭 계산
    
    Args:
        outputs: 모델 출력
        labels: 실제 레이블
    
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