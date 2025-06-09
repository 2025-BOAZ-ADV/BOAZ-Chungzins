"""1. Pretraining 스크립트"""

import os
import argparse
from pathlib import Path
import torch
from importlib import import_module

from data.dataset import CycleDataset
from data.splitter import get_shuffled_filenames, split_cycledataset
from models.backbone import create_backbone
from models.moco import MoCo
from trainers.pretrain import PretrainTrainer
from utils.logger import WandbLogger

def parse_args():
    parser = argparse.ArgumentParser(description='STEP 1. Pretraining')
    parser.add_argument('--exp', type=str, required=True,
                        help='실험 설정 파일 (experiments 폴더 내 파일명)')
    parser.add_argument('--resume', type=str,
                        help='체크포인트에서 재시작')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 현재 프로젝트 루트 디렉토리 설정
    project_root = Path(__file__).parent.parent
    
    # 실험 설정 로드
    exp_module = import_module(f'scripts.experiments.{args.exp}')
    config = exp_module.ExperimentConfig
    ssl_config = config.ssl()  # 인스턴스화
    
    # 디렉토리 생성
    checkpoints_dir = project_root / 'checkpoints' / args.exp
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    wandb_dir = project_root / 'wandb_logs'
    wandb_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb 초기화
    logger = WandbLogger(
        project_name=f"{config.wandb_project}-pretrain",
        config=vars(ssl_config),
        entity=config.wandb_entity
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
        use_cache=False,    
        save_cache=True
    )

    # train data의 일부를 가져와 사전훈련용 데이터셋 구축
    pretrain_filename_list = get_shuffled_filenames(
        metadata_path=str(metadata_path),
        option="pretrain",
        split_ratio=config.split_ratio,
        seed=config.seed
    )
    pretrain_dataset = split_cycledataset(
        train_dataset,
        pretrain_filename_list,
        seed=config.seed
    )

    # DataLoader 생성
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_dataset,
        batch_size=ssl_config.batch_size,
        shuffle=False,      # 추후 개선할 부분, 이미 dataset이 한번 셔플된 상태
        num_workers=0 if device.type == 'cpu' else ssl_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True      # 추후 개선할 부분
    )
    
    # Multi-label MoCo 모델 생성
    model = MoCo(
        base_encoder=create_backbone
    ).to(device)
    
    # 체크포인트에서 재시작
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Trainer 생성
    trainer = PretrainTrainer(
        model=model,
        augmentations=ssl_config.augmentations,
        train_loader=pretrain_loader,
        device=device,
        config=ssl_config,
        logger=logger
    )
    
    # 학습 실행
    history = trainer.train(
        epochs=ssl_config.epochs - start_epoch,
        save_path=str(checkpoints_dir)
    )
    
    # wandb 종료
    logger.finish()

if __name__ == "__main__":
    main()