import numpy as np
import torch
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def get_confusion_matrix_for_multi_label(all_labels, all_preds):
    """각 label에 대한 혼동 행렬 계산
    
    Args:
        all_labels: 정답 multi_label
        all_preds: 정답 multi_label
    
    Returns:
        각 단일 label의 혼동 행렬 (Dict)
    """
    
    confusion_matrices = {
        'Crackle': confusion_matrix(all_labels[:, 0], all_preds[:, 0]),
        'Wheeze': confusion_matrix(all_labels[:, 1], all_preds[:, 1])
    }
    
    return confusion_matrices

def log_confusion_matrix_for_multi_label(confusion_matrices, logger=None):
    """각 label에 대한 혼동 행렬을 로그
    
    Args:
        confusion_matrices: 각 단일 label의 혼동 행렬 (Dict)
        logger: WandbLogger
    """
    label_names = ['Crackle', 'Wheeze']

    for label_name in label_names:
        # 2x2 matrix
        conf_matrix = confusion_matrices[label_name]

        # 혼동 행렬 정규화
        normalized_conf_matrix = np.array(conf_matrix) / np.array(conf_matrix).sum()

        # 정규화된 혼동 행렬 로깅 (소숫점 둘째 자리에서 반올림)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['0', '1'], yticklabels=['0', '1'], ax=ax)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix of {label_name}')

        # Positive: 1 / Negative: 0
        TP = conf_matrix[1, 1]          # 양성 중에 양성으로 예측
        FN = conf_matrix[1, 0]          # 양성인데 음성으로 예측
        FP = conf_matrix[0, 1]          # 음성인데 양성으로 예측
        TN = conf_matrix[0, 0]          # 음성인데 양성으로 예측

        sens = TP / (TP + FN + 1e-6)    # 민감도
        spec = TN / (TN + FP + 1e-6)    # 특이도

        plt.text(
            0.99, 0.35,  # 우하단 (x=99%, y=35%) 위치
            f"Sensitivity: {sens*100:.2f}\nSpecificity: {spec*100:.2f}\nICBHI Score: {100*(sens+spec)/2:.2f}",
            ha='right', va='bottom',
            transform=plt.gca().transAxes,  # 축 기준 좌표로 해석
            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )

        plt.tight_layout()

        # logging
        if logger:
            logger.log({f"2x2 Confusion Matrix of {label_name}": wandb.Image(fig)})

            logger.log({
                f"Metrics/[{label_name}] Sensitivity": sens,
                f"Metrics/[{label_name}] Specificity": spec,
                f"Metrics/[{label_name}] ICBHI Score": (sens+spec)/2
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
        all_labels: 실제 레이블
        all_preds: 모델 출력
    
    Returns:
        클래스별 메트릭 딕셔너리
    """
    
    all_labels_cls = convert_to_multi_class(all_labels)
    all_preds_cls = convert_to_multi_class(all_preds)

    # 4x4 matrix
    conf_matrix = confusion_matrix(all_labels_cls, all_preds_cls, labels=[0,1,2,3])

    # 혼동 행렬 정규화
    normalized_conf_matrix = np.array(conf_matrix) / np.array(conf_matrix).sum()

    # Positive: 1,2,3 / Negative: 0
    TP = conf_matrix[1:, 1:].sum()    # 양성 중에 양성으로 예측
    FN = conf_matrix[1:, 0].sum()     # 양성인데 음성으로 예측
    FP = conf_matrix[0, 1:].sum()     # 음성인데 양성으로 예측
    TN = conf_matrix[0, 0]            # 음성인데 양성으로 예측

    sens = TP / (TP + FN + 1e-6)    # 민감도
    spec = TN / (TN + FP + 1e-6)    # 특이도

    print("4x4 Confusion Matrix:\n", normalized_conf_matrix)
    print(f"[4Class] Sens: {sens:.4f}, Spec: {spec:.4f}, ICBHI Score: {(sens+spec)/2:.4f}")

    return normalized_conf_matrix, sens, spec

def log_confusion_matrix_for_multi_class(normalized_conf_matrix, sens, spec, logger=None):

    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']

    # 정규화된 혼동 행렬 로깅 (소숫점 둘째 자리에서 반올림)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
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

    # logging
    if logger:
        logger.log({"Multi-Class 4x4 Confusion Matrix": wandb.Image(fig)})

        logger.log({
            "Metrics/[4Class] Sensitivity": sens,
            "Metrics/[4Class] Specificity": spec,
            "Metrics/[4Class] ICHBI Score": (sens+spec)/2
        })

    plt.close(fig)