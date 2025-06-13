import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any, Union
from sklearn.metrics import f1_score

from models.classifier import LungSoundClassifier
from utils.logger import get_timestamp, WandbLogger

class FinetuneTrainer:
    def __init__(
        self,
        model: LungSoundClassifier,
        device: torch.device,
        config: Any,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        logger: WandbLogger = None
    ):
        self.model = model
        self.device = device
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        
        # Binary Cross Entropy for multi-label classification
        # --------------- 나중에 Focal Loss로 수정할 수 있음! ---------------
        self.criterion = nn.BCEWithLogitsLoss()
        
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

        # Validation Loader 입력 여부 출력
        if not isinstance(self.val_loader, DataLoader):
            self.is_val_loader = False
            print("[INFO] Validation set이 없으므로 모든 데이터를 Training에 사용합니다.")

    def train_epoch(self) -> Tuple[float, float]:
        """한 epoch 학습
        
        Returns:
            loss, f1 score
        """
        # 훈련 모드 전환 (Dropout, Batchnorm 적용)
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
            
            # loss 합산 및 정답/예측 label 저장
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(multi_label.cpu().tolist())
            
            # progress bar 업데이트
            progress_bar.set_postfix({'Loss': loss.item()})

        print(all_preds)
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return epoch_loss, epoch_f1
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """fine-tuning 과정에서 validation 수행
        
        Returns:
            validation loss, f1 score
        """
        # 평가 모드 전환 (Dropout, BatchNorm 스킵)
        self.model.eval()

        # Validation Loader를 입력받지 않을 경우 Validation 스킵
        if self.is_val_loader == False:
            return -1.0, -1.0

        total_loss = 0
        all_preds = []
        all_labels = []
        
        for mel, multi_label, _ in self.val_loader:
            mel, multi_label = mel.to(self.device), multi_label.to(self.device)
            
            outputs = self.model(mel)
            loss = self.criterion(outputs, multi_label)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(multi_label.cpu().tolist())
        
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
        best_train_f1 = 0.0
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
                if self.is_val_loader == False:
                    self.logger.log({
                        'Finetune/epoch': epoch,
                        'Finetune/train_loss': train_loss,
                        'Finetune/train_f1': train_f1,
                        'Finetune/learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                else:
                    self.logger.log({
                        'Finetune/epoch': epoch,
                        'Finetune/train_loss': train_loss,
                        'Finetune/train_f1': train_f1,
                        'Finetune/val_loss': val_loss,
                        'Finetune/val_f1': val_f1,
                        'Finetune/learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                
            if save_path:
                # Validation Loader를 입력받지 않은 경우, Train F1 Score 기준으로 best model 저장
                if self.is_val_loader == False:
                    if train_f1 > best_train_f1:
                        best_train_f1 = train_f1
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'best_val_f1': best_val_f1,
                        }, f'{save_path}/best_finetuned_model_{get_timestamp()}.pth')

                # Validation Loader를 입력받은 경우, Validation F1 Score와 비교하여 best model 저장
                elif val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_val_f1': best_val_f1,
                    }, f'{save_path}/best_finetuned_model_{get_timestamp()}.pth')
            
            # epoch 당 loss, f1 출력
            print(f'Epoch {epoch}:')
            print(f'Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}')

            if self.is_val_loader == True:
                print(f'Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}')
        
        return history
