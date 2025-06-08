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

def parse_args():
    parser = argparse.ArgumentParser(description='STEP 3. Test')
    parser.add_argument('--exp', type=str, required=True,
                        help='실험 설정 파일 (experiments 폴더 내 파일명)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='데이터 디렉토리 경로')
    parser.add_argument('--ssl-checkpoint', type=str, required=True,
                        help='SSL 모델 체크포인트 경로')
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
        project_name=f"{config.wandb_project}-test",
        entity=config.wandb_entity,
        config=vars(config.finetune)
    )
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # test data로 CycleDataset 생성
    test_dataset = CycleDataset(
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        option="test",
        target_sr=ssl_config.target_sr,
        target_sec=ssl_config.target_sec,
        frame_size=ssl.config.frame_size,
        hop_length=ssl.config.hop_length,
        n_mels=ssl.config.n_mels,
        use_cache=False,    # 추후 True로 바꾸기
        save_cache=True
    )
    
    # DataLoader 생성
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=ssl_config.batch_size,
        shuffle=False,      # 추후 개선할 부분, 이미 dataset이 한번 셔플된 상태
        num_workers=0 if device.type == 'cpu' else ssl_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True      # 추후 개선할 부분
    )

    # 모델 생성 (Classifier까지 훈련된 것을 가져옴)
    model = create_classifier(
        ssl_checkpoint=args.ssl_checkpoint,
        num_classes=2,  # Crackle, Wheeze
        freeze_backbone=config.finetune.freeze_backbone,
        dropout_rate=config.finetune.dropout_rate
    ).to(device)
    
    # 평가 모드 전환
    model.eval()

    # Trainer 생성
    tester = TestRunner(
        model=model,
        test_loader=test_loader
    )
    
    # 성능 계산
    avg_results, all_labels, all_preds = tester.test()

    # 각 label의 2x2 Confusion matrix wandb 이미지와 평균 성능 로그
    log_confusion_matrix_for_multi_label(get_confusion_matrix_for_multi_label(all_labels, all_preds), avg_results, logger)

    # multi-class로 변환하여 성능 출력
    conf_matrix, sens, spec = get_confusion_matrix_for_multi_class(all_labels, all_preds)

    # 2x2 Confusion matrix wandb 이미지 로그
    log_confusion_matrix_for_multi_class(conf_matrix, sens, spec, logger):
    
    # wandb 종료
    logger.finish()

if __name__ == "__main__":
    main()
