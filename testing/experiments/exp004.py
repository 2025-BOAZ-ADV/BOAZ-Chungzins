'''
Shell Script 자동화 실험

bash에서 명령어 수행 순서:  (1은 한번만 하면 됨)
1) chmod +x ./testing/experiments/run_sweep.sh
2) ./testing/experiments/run_sweep.sh

아직 실행이 안 됩니다.
앞으로 할 순서
1. CLI -> class를 불러와 오버라이드
2. ClassifierConfig -> nn.을 파라미터로 지정하지 않기
3. 실험 이름은 그냥 각 trainer.py에서 짓도록
'''

import argparse, json, os, textwrap
from importlib import import_module
from pathlib import Path

from testing.config.combined_config import CombinedConfig
from utils.logger import get_timestamp

# ---------- Helper functions ----------
def to_dict(obj):
    """클래스 → 딕셔너리 변환"""
    return obj.__dict__

def merge(*dicts):
    """모든 Config 클래스를 통합"""
    merged = {}
    for d in dicts:
        if d:  # None이면 건너뜀
            merged.update({k: v for k, v in d.items() if v is not None})
    return merged

def to_serializable_dict(cfg: dict):
    """json 파일을 저장할 때, 저장할 수 없는 항목은 스킵"""
    result = {}
    for k, v in cfg.items():
        try:
            json.dumps(v) 
            result[k] = v
        except TypeError:
            pass
    return result

def log_results(cfg, tag, log_line):
    # 프로젝트 최상위 경로 설정
    project_root = Path(__file__).parent.parent

    # json 파일을 저장할 경로 설정
    out_dir = project_root / 'experiments' / 'details' / f"{get_timestamp()}" / f"{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 실험에 사용한 모든 파라미터들을 json 파일로 저장
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(to_serializable_dict(cfg), f, indent=2)

    # 변경한 파라미터 정보를 log 파일로 저장
    print(f"{log_line}\n")
    with open(os.path.join(out_dir, "run.log"), "a") as lf:
        lf.write(log_line)

# --------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 0. CLI 덮어쓰기용 파라미터
    parser.add_argument("--lambda_bce", type=float, help="override lambda_bce")
    parser.add_argument("--target_sec", type=int, help="override target_sec")
    args = parser.parse_args()

    # 1. 모든 파라미터 병합
    cfg = to_dict(CombinedConfig(backbone='resnet', method='mls'))
    cli   = {
        "lambda_bce": args.lambda_bce,
        "target_sec": args.target_sec,
    }

    cfg = merge(cfg, cli)

    # 2. 파라미터 정보 저장
    tag = f"ld{cfg['lambda_bce']}_sec{cfg['target_sec']}"
    log_line = textwrap.dedent(f"""
        Changed Params:
        - ld={cfg['lambda_bce']}
        - sec={cfg['target_sec']}

    """).strip()

    log_results(cfg, tag, log_line)

    # 3. 훈련 수행
    experiment = import_module(f'testing.test_scripts.run_pretrain')
    experiment.main(cfg)

if __name__ == "__main__":
    main()
