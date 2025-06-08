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

class FinetuneTrainer:
    def __init__(
        self,
        model: LungSoundClassifier,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        config: Config,
        logger: WandbLogger = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logger = logger
        
        # Binary Cross Entropy for multi-label classification
        # --------------- 나중에 Focal Loss로 수정할 수 있음 ---------------
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.finetune_lr,
            weight_decay=config.finetune_weight_decay
        )
        
        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

    def train_epoch(self) -> Tuple[float, float]:
        """한 epoch 학습
        
        Returns:
            loss, f1 score
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader)
        for mel, multi_label, _ in progress_bar:
            mel, multi_label = mel.to(self.device), multi_label.to(self.device)
            
            # forward pass
            outputs = self.model(mel)
            loss = self.criterion(outputs, multi_label)
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 통계
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(multi_label.cpu().numpy())
            
            # progress bar 업데이트
            progress_bar.set_postfix({'Loss': loss.item()})
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return epoch_loss, epoch_f1
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """검증 수행
        
        Returns:
            validation loss, f1 score
        """
        # Validation Set이 없을 경우 아래처럼 작성하면 Validation을 건너뛰고 Train loss만 계산됨
        if not self.val_loader:
            print("=========== No Validation Loader ===========")
            return 0.0, 0.0
            
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for mel, multi_label, _ in self.val_loader:
            mel, multi_label = mel.to(self.device), multi_label.to(self.device)
            
            outputs = self.model(mel)
            loss = self.criterion(outputs, multi_label)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(multi_label.cpu().numpy())
        
        val_loss = total_loss / len(self.val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return val_loss, val_f1
    
    def train(self, epochs: int, save_path: str = None) -> Dict:
        """전체 fine-tuning 과정
        
        Args:
            epochs: 전체 epoch 수
            save_path: 체크포인트 저장 경로
        
        Returns:
            학습 히스토리
        """
        history = {
            'train_loss': [], 'train_f1': [],
            'val_loss': [], 'val_f1': []
        }
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            # train
            train_loss, train_f1 = self.train_epoch()
            
            # validate
            val_loss, val_f1 = self.validate()
            
            # history 업데이트
            history['train_loss'].append(train_loss)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            
            # scheduler step
            self.scheduler.step()
            
            # logging
            if self.logger:
                self.logger.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_f1': train_f1,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # save best model
            if save_path and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_f1': best_val_f1,
                }, f'{save_path}/best_model.pth')
            
            print(f'Epoch {epoch}:')
            print(f'Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}')
        
        return history
