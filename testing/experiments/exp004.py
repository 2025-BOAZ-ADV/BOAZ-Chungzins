'''Shell Script 자동화 실험'''

import argparse, json, os, textwrap
from importlib import import_module
from pathlib import Path

from testing.config.data.preprocess import PreprocessConfig
from testing.config.models.methods.mls_moco import MLSMocoConfig
from testing.config.trainers.pretrain import PretrainConfig
from testing.config.data.augmentations import AugmentationConfig

from utils.logger import get_timestamp

# ---------- Helper functions ----------
def to_dict(obj):
    """클래스 → 딕셔너리 변환"""
    return obj.__dict__

def merge(*dicts):
    merged = {}
    for d in dicts:
        if d:  # None이면 건너뜀
            merged.update({k: v for k, v in d.items() if v is not None})
    return merged
# --------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # CLI 덮어쓰기용 파라미터
    parser.add_argument("--lambda_bce", type=float, help="override lambda_bce")
    parser.add_argument("--target_sec", type=int, help="override target_sec")
    args = parser.parse_args()

    # 현재 프로젝트 루트 디렉토리 설정
    project_root = Path(__file__).parent.parent

    # 1) 각 설정 import → dict 변환
    cfg1 = to_dict(PreprocessConfig())
    cfg2 = to_dict(MLSMocoConfig())
    cfg3 = to_dict(PretrainConfig())
    cfg4 = to_dict(AugmentationConfig())

    cli   = {
        "lambda_bce": args.lambda_bce,
        "target_sec": args.target_sec,
    }

    # 2) 병합 (CLI 값이 가장 마지막 → 가장 높은 우선순위)
    cfg = merge(cfg1, cfg2, cfg3, cfg4, cli)

    # 3) 결과 저장
    ts  = get_timestamp()
    tag = f"ld{cfg['lambda_bce']}_sec{cfg['target_sec']}"

    out_dir = project_root / 'experiments' / 'details' / f"{ts}" / f"{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # 4) 콘솔 & 파일 로그
    log_line = textwrap.dedent(f"""
                               
        Changed Params:
        - ld={cfg['lambda_bce']}
        - sec={cfg['target_sec']}

    """).strip()
    print(f"{log_line}\n")

    with open(os.path.join(out_dir, "run.log"), "a") as lf:
        lf.write(log_line)

    # 5) 훈련 수행
    experiment = import_module(f'testing.experiments.exp003')
    experiment.main(cfg)
    # -----------------------------------

if __name__ == "__main__":
    main()

# bash에서 명령어 수행:  (1은 한번만 하면 됨)
# 1) chmod +x ./testing/experiments/run_sweep.sh
# 2) ./testing/experiments/run_sweep.sh