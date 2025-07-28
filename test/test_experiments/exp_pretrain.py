'''
Shell Script 자동화 실험

bash에서 명령어 수행 순서:  (1은 한번만 하면 됨)
1) chmod +x ./test_experiments/run_sweep.sh
2) ./test_experiments/run_sweep.sh

'''

import argparse, json, textwrap
from importlib import import_module
from pathlib import Path

from test_config.combined_config import CombinedConfig
from utils.logger import get_timestamp


class ExperimentConfig(CombinedConfig):
    def __init__(self, backbone="resnet", method="mls", **overrides):
        super().__init__(backbone=backbone, method=method)

        for k, v in overrides.items():
            if v is not None:
                setattr(self, k, v)


def to_serializable_dict(d: dict):
    """JSON 직렬화 가능한 항목만 필터링"""
    result = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            result[k] = v
        except TypeError:
            pass
    return result


def log_results(cfg, tag, log_line):
    """실험에 사용한 파라미터 정보를 로깅"""
    project_root = Path(__file__).parent.parent
    out_dir = project_root / 'test_experiments' / 'details' / f"{get_timestamp()}" / f"{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 실험에 사용한 모든 파라미터들을 json 파일로 저장
    with open(out_dir / "config.json", "w") as f:
        json.dump(to_serializable_dict(cfg.__dict__), f, indent=2)

    # 실험에서 변경해준 파라미터들을 log 파일로 저장
    print(f"{log_line}\n")
    with open(out_dir / "run.log", "a") as lf:
        lf.write(log_line)


def main():
    """변경할 파라미터를 shell script로부터 입력받아 실험 수행"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_bce", type=float)
    parser.add_argument("--target_sec", type=int)
    args = parser.parse_args()

    # CLI 입력을 dict로 변환
    cli_overrides = vars(args)

    # config 객체 생성 + CLI 파라미터로 일부 덮어쓰기
    cfg = ExperimentConfig(**cli_overrides)

    # 로깅용 태그 및 메시지 구성
    tag = f"ld{cfg.lambda_bce}_sec{cfg.target_sec}"
    log_line = textwrap.dedent(f"""
        Changed Params:
        - ld={cfg.lambda_bce}
        - sec={cfg.target_sec}
    """).strip()

    # 로그 저장
    log_results(cfg, tag, log_line)

    # 실험 모듈 실행
    experiment = import_module(f'test_scripts.run_pretrain')
    experiment.main(cfg)

if __name__ == "__main__":
    main()
