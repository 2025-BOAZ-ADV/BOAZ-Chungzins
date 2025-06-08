import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple

from models.moco import MoCo
from data.augmentation import create_augmenter, apply_spec_augment
from utils.logger import WandbLogger
from config.config import Config

class PretrainTrainer:
    def __init__(
        self,
        model: MoCo,
        train_loader: DataLoader,
        device: torch.device,
        config: Config,
        logger: WandbLogger = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.config = config
        self.logger = logger
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """한 epoch 학습
        
        Args:
            epoch: 현재 epoch
        
        Returns:
            loss, accuracy
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (mel, _, _) in enumerate(progress_bar):
            # Augmentation 생성 인스턴스 정의
            augmenter = create_augmenter(self.config, self.augmentations)

            # 두 개의 augmentation 적용
            aug1, aug2 = augmenter.generate_views(mel)[0], augmenter.generate_views(mel)[1]
            aug1, aug2 = aug1.to(self.device), aug2.to(self.device)
            
            # forward pass
            loss, logits, labels = self.model(aug1, aug2, epoch)
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 통계
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            # progress bar 업데이트
            progress_bar.set_postfix({
                'Loss': total_loss / (batch_idx + 1),
                'Acc': 100. * correct / total
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs: int, save_path: str = None) -> Dict:
        """전체 학습 과정
        
        Args:
            epochs: 전체 epoch 수
            save_path: 체크포인트 저장 경로
        
        Returns:
            학습 히스토리
        """
        history = {'loss': [], 'acc': []}
        
        for epoch in range(epochs):
            # train epoch
            loss, acc = self.train_epoch(epoch)
            history['loss'].append(loss)
            history['acc'].append(acc)
            
            # scheduler step
            self.scheduler.step()
            
            # logging
            if self.logger:
                self.logger.log({
                    'epoch': epoch,
                    'train_loss': loss,
                    'train_acc': acc,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # save checkpoint
            if save_path and (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': loss,
                }, f'{save_path}/best_pretrained_model.pth')
        
        return history
