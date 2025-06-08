import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from sklearn.metrics import f1_score

from models.classifier import LungSoundClassifier
from utils.logger import WandbLogger
from config.config import Config

class TestRunner:
    def __init__(
        self,
        model: LungSoundClassifier,
        test_loader: DataLoader
    ):
        self.model = model
        self.train_loader = train_loader

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
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, _ in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(multi_label.cpu().numpy())

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

        # sens, spec, socre의 평균값 계산
        avg_results = {
            key: sum([results[lbl][key] for lbl in results]) / 2
            for key in ['sensitivity', 'specificity', 'ICBHI score']
        }

        # label별 성능 출력 (2x2 confusion matrix 기준)
        for lbl in ['Crackle', 'Wheeze']:
            r = results[lbl]
            print(f"  [{lbl}] Sens: {r['sensitivity']:.4f}, Spec: {r['specificity']:.4f}, ICBHI Score: {r['ICBHI score']:.4f}")

        # label별 평균 성능 출력
        print(f"  [Average] Sens: {avg_results['sensitivity']:.4f}, Spec: {avg_results['specificity']:.4f}, ICBHI Score: {avg_results['ICBHI score']:.4f}")

        return avg_results, all_labels, all_preds