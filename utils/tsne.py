import torch
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE

from utils.logger import get_timestamp, WandbLogger

# t-SNE를 위한 feature 추출 함수
@torch.no_grad()
def extract_features(encoder, dataloader, device):
    all_features = []
    all_labels = []

    for mel, multi_label, _ in tqdm(dataloader, desc="Extracting features..."):
        mel = mel.to(device)
        out = encoder(mel)
        out = torch.nn.functional.normalize(out, dim=1)  # L2 정규화
        all_features.append(out.cpu())
        all_labels.append(multi_label.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()    # multi-label 형태로 저장

    return all_features, all_labels

# t-SNE 시각화 수행 함수
def plot_tsne(
        all_features,
        all_labels,
        logger: WandbLogger,
        sens: Optional[float] = None,
        spec: Optional[float] = None
    ): 

    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(all_features)

    # 선택한 라벨만 추출
    for i, option in enumerate(['Crackle', 'Wheeze']):
        labels = all_labels[:, 0] if option == "Crackle" else all_labels[:, 1]
        label_names = ["Normal", option]

        # 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        for lbl, name in enumerate(label_names):
            idx = (labels == lbl)
            ax.scatter(reduced[idx, 0], reduced[idx, 1], label=name, alpha=0.6, edgecolors='k')

        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_title(f't-SNE Visualization of {option}')
        ax.legend()
        ax.grid(True)

        # 성능 지표 텍스트 표시
        if all(v is not None for v in (sens, spec)):
            ax.text(
                0.95, 0.1,
                f"Sensitivity: {sens*100:.2f}\nSpecificity: {spec*100:.2f}\nICBHI Score: {(sens + spec)*50:.2f}",
                ha='right', va='bottom',
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )
            
        plt.tight_layout()
        plt.show()

        # logging
        if logger:
            logger.log({f't-SNE Visualization of {option} ({get_timestamp()})': wandb.Image(fig)})