import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any, Union
from sklearn.metrics import confusion_matrix, f1_score

from models.classifier import LungSoundClassifier

class TestRunner:
    def __init__(
        self,
        model: LungSoundClassifier,
        device: torch.device,
        test_loader: DataLoader,
    ):
        self.model = model
        self.device = device
        self.test_loader = test_loader

    @torch.no_grad
    def test(self):
        """
        Test data를 입력하여 label별 sensitivity, specificity, ICBHI score 계산
        Returns:
            avg_results: 각 metric의 평균
            results: 각 클래스별 metric dict
            all_labels: 정답 multi_label (numpy array)
            all_preds: 예측 multi_label (numpy array)
        """
        # 평가 모드 전환 (Dropout, BatchNorm 스킵)
        self.model.eval()

        all_preds = []
        all_labels = []

        progress_bar = tqdm(self.test_loader)
        with torch.no_grad():
            for mel, multi_label, _ in progress_bar:
                mel, multi_label = mel.to(self.device), multi_label.to(self.device)

                outputs = self.model(mel)
                preds = (torch.sigmoid(outputs) > 0.5).float()

                all_preds.append(preds.cpu())
                all_labels.append(multi_label.cpu())
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        print(f"[DEBUG] 10 Predictions: {all_preds[0:10]}")

        # 개별 label별 성능 계산
        results = {}
        for i, lbl_name in enumerate(['Crackle', 'Wheeze']):
            y_true = all_labels[:, i]
            y_pred = all_preds[:, i]

            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel()
            sens = TP / (TP + FN + 1e-6)
            spec = TN / (TN + FP + 1e-6)
            score = (sens + spec) / 2

            results[lbl_name] = {
                'sensitivity': sens,
                'specificity': spec,
                'ICBHI score': score
            }

        # label별 성능 출력 (2x2 confusion matrix 기준)
        for lbl in ['Crackle', 'Wheeze']:
            r = results[lbl]
            print(f"  [{lbl}] Sens: {r['sensitivity']:.4f}, Spec: {r['specificity']:.4f}, ICBHI Score: {r['ICBHI score']:.4f}")

        return all_labels, all_preds