"""3. Test 스크립트"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from importlib import import_module

from data.dataset import CycleDataset
from models.classifier import create_classifier
from trainers.test import TestRunner
from utils.logger import get_timestamp, WandbLogger
from utils.metrics import get_confusion_matrix_for_multi_label, log_confusion_matrix_for_multi_label
from utils.metrics import get_confusion_matrix_for_multi_class, log_confusion_matrix_for_multi_class
from utils.tsne import extract_features, plot_tsne

def main(cfg):

    # 현재 프로젝트 루트 디렉토리 설정
    project_root = Path(__file__).parent.parent
    
    # wandb 실험 이름
    experiment_name = (
        f"test-{cfg.batch_size}bs-{cfg.target_sr//1000}kHz-"
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

    # t-SNE 결과 저장 폴더
    out_dir = project_root / 'pictures' / 'tsne_results' / str(get_timestamp())
    
    # 데이터셋 경로 설정
    data_path = project_root / 'data' / 'raw'
    metadata_path = project_root / 'data' / 'metadata'
    
    # test data로 CycleDataset 생성
    test_dataset = CycleDataset(
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        option="test",
        target_sr=cfg.target_sr,
        target_sec=cfg.target_sec,
        frame_size=cfg.frame_size,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        use_cache=cfg.use_cache,
        save_cache=cfg.save_cache
    )
    
    # DataLoader 생성
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,      # 추후 개선할 부분, 이미 dataset이 한번 셔플된 상태
        num_workers=0 if device.type == 'cpu' else cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True      # 추후 개선할 부분
    )

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

    # 모델 생성 (분류기까지 훈련된 것의 경로를 가져옴)
    model = create_classifier(
        checkpoint_path=args.ssl_checkpoint,
        backbone_config=cfg,
        classifier=classifier,
        freeze_encoder=cfg.freeze_encoder
    ).to(device)

    # Trainer 생성 (내부에서 평가모드로 전환함)
    tester = TestRunner(
        model=model,
        device=device,
        test_loader=test_loader
    )
    
    # 성능 평가 수행
    all_labels, all_preds = tester.test()

    # 각 label의 2x2 Confusion matrix wandb 이미지와 성능 로그
    log_confusion_matrix_for_multi_label(get_confusion_matrix_for_multi_label(all_labels, all_preds), logger)

    # 4x4 Confusion matrix wandb 이미지와 성능 로그
    conf_matrix, sens, spec = get_confusion_matrix_for_multi_class(all_labels, all_preds)
    log_confusion_matrix_for_multi_class(conf_matrix, sens, spec, logger)

    # t-SNE 시각화
    all_features, all_labels = extract_features(model.encoder, test_loader, device, dim_mlp=cfg.dim_mlp)
    plot_tsne(all_features, all_labels, logger, sens=sens, spec=spec, save_dir=out_dir)
    
    # wandb 종료
    logger.finish()

if __name__ == "__main__":
    main()
