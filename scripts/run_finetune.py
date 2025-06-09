"""2. Fine-tuning 스크립트"""

import os
import argparse
from pathlib import Path
import torch
from importlib import import_module
from sklearn.model_selection import train_test_split

from data.dataset import CycleDataset
from data.splitter import get_shuffled_filenames, split_cycledataset, create_dataloaders
from models.classifier import create_classifier
from trainers.finetune import FinetuneTrainer
from utils.logger import WandbLogger

def parse_args():
    parser = argparse.ArgumentParser(description='STEP 2. Fine-tuning')
    parser.add_argument('--exp', type=str, required=True,
                        help='실험 설정 파일 (experiments 폴더 내 파일명)')
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
    checkpoints_dir = project_root / 'checkpoints' / args.exp
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb 초기화
    logger = WandbLogger(
        project_name=f"{config.wandb_project}-finetune",
        entity=config.wandb_entity,
        config=vars(config.finetune)
    )
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    data_path = project_root / 'data' / 'raw'
    metadata_path = project_root / 'data' / 'metadata'
    
    # train data로 CycleDataset 생성
    train_dataset = CycleDataset(
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        option="train",
        target_sr=ssl_config.target_sr,
        target_sec=ssl_config.target_sec,
        frame_size=ssl_config.frame_size,
        hop_length=ssl_config.hop_length,
        n_mels=ssl_config.n_mels,
        use_cache=False,    # 추후 True로 바꾸기
        save_cache=True
    )
    
    # train data의 일부를 가져와 파인튜닝용 데이터셋 구축
    finetune_filename_list = get_shuffled_filenames(
        metadata_path=str(metadata_path),
        option="finetune",
        split_ratio=config.split_ratio,
        seed=config.seed
    )
    finetune_dataset = split_cycledataset(
        train_dataset,
        finetune_filename_list,
        seed=config.seed
    )
    
    ##### 파인튜닝용 데이터셋 내에서 다시 train-validation split #####
    ##### 현재는 하지 않습니다. 테스트도 아직 안 한 상태 (기본값=False) #####
    if allow_val == True: 
        # train-val filename split
        train_file_list, val_file_list = train_test_split(
            finetune_filename_list,
            test_size=config.val_ratio,
            random_state=42
        )
    
        # train-val idx split
        train_indices = [i for i, (_, _, meta) in enumerate(train_dataset) 
                        if meta[0] in train_file_list]
        val_indices = [i for i, (_, _, meta) in enumerate(train_dataset) 
                    if meta[0] in val_file_list]
        
        # DataLoader 생성
        dataloaders = create_dataloaders(
            train_dataset=torch.utils.data.Subset(train_dataset, train_indices),
            val_dataset=torch.utils.data.Subset(train_dataset, val_indices),
            batch_size=config.finetune.batch_size,
            num_workers=config.finetune.num_workers
        )

        train_loader = dataloaders['train']
        val_loader = dataloaders['val']

    else:
        train_loader = torch.utils.data.DataLoader(
            finetune_dataset,
            batch_size=config.finetune.batch_size,
            shuffle=False,      # 추후 개선할 부분, 이미 dataset이 한번 셔플된 상태
            num_workers=0 if device.type == 'cpu' else config.finetune.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True      # 추후 개선할 부분
        )

        # Validation을 하지 않을 경우 val_loader는 None으로 지정
        val_loader = None
    
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
    
    # Trainer 생성
    trainer = FinetuneTrainer(
        model=model,
        device=device,
        config=config.finetune,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger
    )
    
    # 학습 실행
    history = trainer.train(
        epochs=config.finetune.epochs - start_epoch,
        save_path=str(checkpoints_dir)
    )
    
    # wandb 종료
    logger.finish()

if __name__ == "__main__":
    main()
