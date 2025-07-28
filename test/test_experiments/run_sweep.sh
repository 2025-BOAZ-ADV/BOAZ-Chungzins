#!/usr/bin/env bash
# 실행 전 다음 구문을 bash에 입력(최초 한번만 하면 됨): chmod +x ./test_experiments/run_sweep.sh

export PYTHONPATH=$(pwd)  # 프로젝트 루트를 PYTHONPATH에 추가

LD=(0.3 0.5 1.0)     
SEC=(7 8) 

mkdir -p test_experiments/runs
MASTER_LOG="test_experiments/runs/sweep_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S).log"

count=1  # 실험 카운터 초기화

for ld in "${LD[@]}"; do
  for sec in "${SEC[@]}"; do
    # 실험 번호, 현재 파라미터 로깅
    echo "[${count}] $(TZ=Asia/Seoul date +%T) launch ld=$ld sec=$sec" | tee -a "$MASTER_LOG"

    # 현재 파라미터로 실험 수행
    python -u test_experiments/exp_pretrain.py --lambda_bce "$ld" --target_sec "$sec" \
      2>&1 | tee -a "$MASTER_LOG"

    # 구분선 로깅
    printf '=%.0s' {1..60} >> "$MASTER_LOG"
    echo -e "\n" >> "$MASTER_LOG"

    # 카운터 증가
    ((count++))
  done
done

wait
echo "=== sweep finished ===" | tee -a "$MASTER_LOG"