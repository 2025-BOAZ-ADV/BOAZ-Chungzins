import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Union

from models.moco import MoCo
from data.augmentation import create_augmenter
from utils.logger import get_timestamp, WandbLogger

class PretrainTrainer:
    def __init__(
        self,
        model: MoCo,
        augmentations: List[Dict[str, Any]],
        train_loader: DataLoader,
        device: torch.device,
        config: Any,
        logger: WandbLogger = None,
    ):
        self.model = model
        self.augmentations = augmentations
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
            augmenter = create_augmenter(self.augmentations)

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
                    'Pretrain/epoch': epoch,
                    'Pretrain/train_loss': loss,
                    'Pretrain/train_acc': acc,
                    'Pretrain/learning_rate': self.optimizer.param_groups[0]['lr']
                })

            if save_path:
                # 10 미만의 epoch로 실험중일 경우, 마지막 epoch에 파라미터 저장
                if self.config.epochs < 10 and epoch + 1 == self.config.epochs:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': loss,
                    }, f'{save_path}/best_pretrained_model_{get_timestamp()}.pth')

                # 10 이상의 epoch로 실험중일 경우, 100 epoch에 한번씩 파라미터 저장
                elif (epoch + 1) % 100 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': loss,
                    }, f'{save_path}/best_pretrained_model_{get_timestamp()}.pth')
        
        return history
