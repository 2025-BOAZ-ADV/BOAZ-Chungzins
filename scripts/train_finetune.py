"""2. Fine-tuning 스크립트"""

import os
import argparse
from pathlib import Path
import torch
from importlib import import_module
from sklearn.model_selection import train_test_split

from data.dataset import CycleDataset
from data.splitter import split_cycledataset, create_dataloaders
from models.classifier import create_classifier
from trainers.finetune import FinetuneTrainer
from utils.logger import WandbLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tuning Training')
    parser.add_argument('--exp', type=str, required=True,
                        help='실험 설정 파일 (experiments 폴더 내 파일명)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='데이터 디렉토리 경로')
    parser.add_argument('--ssl-checkpoint', type=str, required=True,
                        help='SSL 모델 체크포인트 경로')
    parser.add_argument('--resume', type=str,
                        help='체크포인트에서 재시작')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 실험 설정 로드
    exp_module = import_module(f'scripts.experiments.{args.exp}')
    config = exp_module.ExperimentConfig
    
    # 디렉토리 생성
    out_dir = Path(config.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb 초기화
    logger = WandbLogger(
        project_name=f"{config.wandb_project}-finetune",
        entity=config.wandb_entity,
        config=vars(config.finetune)
    )
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # train data로 CycleDataset 생성
    train_dataset = CycleDataset(
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        option="train",
        target_sr=ssl_config.target_sr,
        target_sec=ssl_config.target_sec,
        frame_size=ssl.config.frame_size,
        hop_length=ssl.config.hop_length,
        n_mels=ssl.config.n_mels,
        use_cache=False,    # 추후 True로 바꾸기
    )
    
    ###############################################33
    # SSL/Finetune 분할
    _, finetune_files = split_by_patient(
        dataset.df,
        test_size=(1 - config.ssl_ratio)
    )
    
    # Train/Val 분할
    train_files, val_files = train_test_split(
        finetune_files,
        test_size=config.val_ratio,
        random_state=42
    )
    
    # 데이터셋 분할
    _, finetune_dataset = split_ssl_finetune(dataset, [], finetune_files)
    train_indices = [i for i, (_, _, meta) in enumerate(finetune_dataset) 
                    if meta[0] in train_files]
    val_indices = [i for i, (_, _, meta) in enumerate(finetune_dataset) 
                  if meta[0] in val_files]
    
    # 데이터로더 생성
    dataloaders = create_dataloaders(
        train_dataset=torch.utils.data.Subset(finetune_dataset, train_indices),
        val_dataset=torch.utils.data.Subset(finetune_dataset, val_indices),
        batch_size=config.finetune.batch_size,
        num_workers=config.finetune.num_workers
    )
    
    # 모델 생성
    model = create_classifier(
        ssl_checkpoint=args.ssl_checkpoint,
        num_classes=2,  # Crackle, Wheeze
        freeze_backbone=config.finetune.freeze_backbone,
        dropout_rate=config.finetune.dropout_rate
    ).to(device)
    
    # 체크포인트에서 재시작
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # 트레이너 생성
    trainer = FinetuneTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=device,
        config=config.finetune,
        logger=logger
    )
    
    # 학습 실행
    history = trainer.train(
        epochs=config.finetune.epochs - start_epoch,
        save_path=str(out_dir)
    )
    
    # wandb 종료
    logger.finish()

if __name__ == "__main__":
    main()
