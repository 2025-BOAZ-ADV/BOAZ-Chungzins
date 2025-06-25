"""2. Fine-tuning 스크립트"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from importlib import import_module
from sklearn.model_selection import train_test_split

from data.dataset import CycleDataset
from data.splitter import get_shuffled_filenames, split_cycledataset, create_dataloaders
from models.classifier import create_classifier
from trainers.finetune import FinetuneTrainer
from utils.logger import get_timestamp, WandbLogger
from utils.tsne import extract_features, plot_tsne

# args.ssl_checkpoint 필요!

def main(cfg):

    # 현재 프로젝트 루트 디렉토리 설정
    project_root = Path(__file__).parent.parent
    
    # 디렉토리 생성
    checkpoints_dir = project_root / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb 실험 이름
    experiment_name = (
        f"fnt-{cfg.batch_size}bs-{cfg.target_sr//1000}kHz-"
        f"{cfg.num_layers}layer-{cfg.dropout_rate}dr-"
        f"{get_timestamp()}"
    )

    # wandb 초기화
    logger = WandbLogger(
        project_name=cfg.wandb_project,
        experiment_name=experiment_name,
        config=vars(cfg),
        entity=cfg.wandb_entity
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
        target_sr=cfg.target_sr,
        target_sec=cfg.target_sec,
        frame_size=cfg.frame_size,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        use_cache=cfg.use_cache,   
        save_cache=cfg.save_cache
    )
    
    # train data의 일부를 가져와 파인튜닝용 데이터셋 구축
    finetune_filename_list = get_shuffled_filenames(
        metadata_path=str(metadata_path),
        option="finetune",
        split_ratio=cfg.split_ratio,
        seed=cfg.seed
    )
    finetune_dataset = split_cycledataset(
        train_dataset,
        finetune_filename_list,
        seed=cfg.seed
    )
    
    ##### 파인튜닝용 데이터셋 내에서 다시 train-validation split #####
    if cfg.allow_val == True: 
        # train-val filename split
        train_file_list, val_file_list = train_test_split(
            finetune_filename_list,
            test_size=cfg.val_ratio,
            random_state=cfg.seed
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
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers
        )

        train_loader = dataloaders['train']
        val_loader = dataloaders['val']

    else:
        train_loader = torch.utils.data.DataLoader(
            finetune_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,      # 추후 개선할 부분, 이미 dataset이 한번 셔플된 상태
            num_workers=0 if device.type == 'cpu' else cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True      # 추후 개선할 부분
        )

        # Validation을 하지 않을 경우 val_loader는 None으로 지정
        val_loader = None
    
    # 분류기 생성
    layers = []
    in_dim = cfg.in_dim

    for out_dim in cfg.layer_dims:
        layers += [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout_rate)
        ]
        in_dim = out_dim

    layers.append(nn.Linear(in_dim, 2))
    classifier = nn.Sequential(*layers)

    # 모델 생성
    model = create_classifier(
        checkpoint_path=args.ssl_checkpoint,
        backbone_config=cfg,
        classifier=classifier,
        freeze_encoder=cfg.freeze_encoder
    ).to(device)
    
    # Trainer 생성
    trainer = FinetuneTrainer(
        model=model,
        device=device,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger
    )
    
    # 학습 실행
    trainer.train(
        epochs=cfg.epochs,
        save_path=str(checkpoints_dir)
    )

    # t-SNE 시각화
    all_features, all_labels = extract_features(model.encoder, train_loader, device)
    plot_tsne(all_features, all_labels, logger)
    
    # wandb 종료
    logger.finish()

if __name__ == "__main__":
    main()
