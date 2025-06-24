"""1. Pretraining 스크립트"""

import os
import argparse
import torch
from torchsummary import summary
from importlib import import_module
from pathlib import Path

from data.dataset import CycleDataset
from data.splitter import get_shuffled_filenames, split_cycledataset
from models.backbone import create_backbone
from models.moco import MoCo
from trainers.pretrain import PretrainTrainer
from utils.logger import get_timestamp, WandbLogger
from utils.tsne import extract_features, plot_tsne

def main(cfg):

    # 현재 프로젝트 루트 디렉토리 설정
    project_root = Path(__file__).parent.parent
    
    # 디렉토리 생성
    checkpoints_dir = project_root / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb 초기화
    logger = WandbLogger(
        project_name=cfg.wandb_project,
        experiment_name=cfg.step1_experiment_name,
        config=vars(cfg),
        entity=cfg.wandb_entity
    )
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # t-SNE 결과 저장 폴더
    out_dir = project_root / 'pictures' / 'tsne_results' / str(get_timestamp())
    
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

    # train data의 일부를 가져와 사전훈련용 데이터셋 구축
    pretrain_filename_list = get_shuffled_filenames(
        metadata_path=str(metadata_path),
        option="pretrain",
        split_ratio=cfg.split_ratio,
        seed=cfg.seed
    )
    pretrain_dataset = split_cycledataset(
        train_dataset=train_dataset,
        filename_list=pretrain_filename_list,
        seed=cfg.seed
    )

    # DataLoader 생성
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,      # 추후 개선할 부분, 이미 dataset이 한번 셔플된 상태
        num_workers=0 if device.type == 'cpu' else cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True      # 추후 개선할 부분
    )
    
    # Multi-label MoCo 모델 생성
    model = MoCo(
        base_encoder=create_backbone,
        config=cfg
    ).to(device)

    # ResNet 내부 구조 출력
    mel_spectrogram_shape = train_dataset[0][0].shape   # mel spectrogram 1개의 shape: (1, 높이, 너비)
    print(summary(create_backbone().to(device), input_size=mel_spectrogram_shape))
    
    # Trainer 생성
    trainer = PretrainTrainer(
        model=model,
        augmentations=cfg.augmentations,
        train_loader=pretrain_loader,
        device=device,
        config=cfg,
        logger=logger
    )
    
    # 학습 실행
    trainer.train(
        epochs=cfg.epochs,
        save_path=str(checkpoints_dir)
    )

    # t-SNE 결과 저장 폴더
    out_dir = project_root / 'tsne_results' / str(get_timestamp())

    # t-SNE 시각화
    all_features, all_labels = extract_features(model.encoder_q, pretrain_loader, device, dim_mlp=cfg.dim_mlp)
    plot_tsne(all_features, all_labels, logger, save_dir=out_dir)

    # wandb 종료
    logger.finish()

if __name__ == "__main__":
    main()