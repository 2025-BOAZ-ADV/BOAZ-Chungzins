"""t-SNE를 다양하게 실험하는 스크립트"""

import torch
import argparse
from pathlib import Path
from importlib import import_module

from data.dataset import CycleDataset
from data.splitter import get_shuffled_filenames, split_cycledataset
from models.classifier import create_classifier
from utils.tsne import extract_features, plot_tsne
from utils.logger import get_timestamp, WandbLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Analysis by Visualization')
    parser.add_argument('--exp', type=str, required=True,
                        help='실험 설정 파일 (experiments 폴더 내 파일명)')
    parser.add_argument('--ssl-checkpoint', type=str, required=True,
                        help='SSL 모델 체크포인트 경로')
    parser.add_argument('--projection', type=str, choices=['y', 'n'], required=True,
                        help='projector 통과 여부 (y/n)')
    parser.add_argument('--data', type=str, choices=['pretrain', 'test'], required=True,
                        help='시각화할 데이터 선택 (pretrain/test)')
    parser.add_argument('--sens', type=float,
                        help='Sensitivity')
    parser.add_argument('--spec', type=float,
                        help='Specificity')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='perplexity')
    parser.add_argument('--max-iter', type=int, default=300,
                        help='max_iter')
    parser.add_argument('--out-dir', type=str, default='tnse_results',
                        help='결과 저장 디렉토리')
    return parser.parse_args()

def main():
    args = parse_args()

    # 현재 프로젝트 루트 디렉토리 설정
    project_root = Path(__file__).parent.parent

    # 실험 설정 로드
    exp_module = import_module(f'scripts.experiments.{args.exp}')
    exp_cfg = exp_module.ExperimentConfig(str(args.exp))
    ssl_cfg = exp_cfg.ssl
    fnt_cfg = exp_cfg.finetune

    # 결과 저장 디렉토리 생성
    out_dir = Path(args.out_dir) / str(get_timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋 경로 설정
    data_path = project_root / 'data' / 'raw'
    metadata_path = project_root / 'data' / 'metadata'
    
    # wandb 초기화
    logger = WandbLogger(
        project_name=exp_cfg.wandb_project,
        experiment_name=exp_cfg.step3_experiment_name,
        config=vars(fnt_cfg),
        entity=exp_cfg.wandb_entity
    )

    # CycleDataset 생성
    if args.data == 'test':
        dataset = CycleDataset(
            data_path=str(data_path),
            metadata_path=str(metadata_path),
            option="test",
            target_sr=fnt_cfg.target_sr,
            target_sec=fnt_cfg.target_sec,
            frame_size=fnt_cfg.frame_size,
            hop_length=fnt_cfg.hop_length,
            n_mels=fnt_cfg.n_mels,
            use_cache=exp_cfg.use_cache,
            save_cache=exp_cfg.save_cache
        )
    elif args.data == 'pretrain':
        train_dataset = CycleDataset(
            data_path=str(data_path),
            metadata_path=str(metadata_path),
            option="train",
            target_sr=ssl_cfg.target_sr,
            target_sec=ssl_cfg.target_sec,
            frame_size=ssl_cfg.frame_size,
            hop_length=ssl_cfg.hop_length,
            n_mels=ssl_cfg.n_mels,
            use_cache=ssl_cfg.use_cache,    
            save_cache=ssl_cfg.save_cache
        )

        pretrain_filename_list = get_shuffled_filenames(
            metadata_path=str(metadata_path),
            option="pretrain",
            split_ratio=exp_cfg.split_ratio,
            seed=exp_cfg.seed
        )
        dataset = split_cycledataset(
            train_dataset=train_dataset,
            filename_list=pretrain_filename_list,
            seed=exp_cfg.seed
        )

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader 생성
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=fnt_cfg.batch_size,
        shuffle=False,      # 추후 개선할 부분, 이미 dataset이 한번 셔플된 상태
        num_workers=0 if device.type == 'cpu' else fnt_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True      # 추후 개선할 부분
    )

    # 모델 생성 (분류기까지 훈련된 것의 경로를 가져옴)
    model = create_classifier(
        checkpoint_path=args.ssl_checkpoint,
        backbone_config=ssl_cfg,
        classifier=fnt_cfg.classifier,
        freeze_encoder=fnt_cfg.freeze_encoder
    ).to(device)

    if args.projection == 'y':
        dim_mlp = ssl_cfg.dim_mlp
    else:
        dim_mlp = None

    # t-SNE 시각화
    all_features, all_labels = extract_features(model.encoder, data_loader, device, dim_mlp=dim_mlp)
    plot_tsne(all_features, all_labels, logger=logger, sens=None, spec=None, perplexity=args.perplexity, max_iter=args.max_iter)

if __name__ == "__main__":
    main()