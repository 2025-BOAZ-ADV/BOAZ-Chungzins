"""3. Test 스크립트"""

import os
import argparse
from pathlib import Path
import torch
from importlib import import_module
from sklearn.model_selection import train_test_split

from data.dataset import CycleDataset
from models.classifier import create_classifier
from trainers.test import TestRunner
from utils.logger import WandbLogger
from utils.metrics import get_confusion_matrix_for_multi_label, log_confusion_matrix_for_multi_label
from utils.metrics import get_confusion_matrix_for_multi_class, log_confusion_matrix_for_multi_class
from utils.tsne import extract_features, plot_tsne

def parse_args():
    parser = argparse.ArgumentParser(description='STEP 3. Test')
    parser.add_argument('--exp', type=str, required=True,
                        help='실험 설정 파일 (experiments 폴더 내 파일명)')
    parser.add_argument('--ssl-checkpoint', type=str, required=True,
                        help='SSL 모델 체크포인트 경로')
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
    
    # 디렉토리 생성
    out_dir = Path(exp_cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb 초기화
    logger = WandbLogger(
        project_name=exp_cfg.wandb_project,
        experiment_name=exp_cfg.step3_experiment_name,
        config=vars(fnt_cfg),
        entity=exp_cfg.wandb_entity
    )
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 경로 설정
    data_path = project_root / 'data' / 'raw'
    metadata_path = project_root / 'data' / 'metadata'
    
    # test data로 CycleDataset 생성
    test_dataset = CycleDataset(
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
    
    # DataLoader 생성
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
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

    # t-SNE 시각화 (고차원 벡터에 t-SNE 적용하길 원할 경우 dim_mlp=None으로 설정)
    all_features, all_labels = extract_features(model.encoder, test_loader, device, dim_mlp=ssl_cfg.dim_mlp)
    plot_tsne(all_features, all_labels, logger, sens=sens, spec=spec)
    
    # wandb 종료
    logger.finish()

if __name__ == "__main__":
    main()
