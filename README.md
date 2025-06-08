# 프로젝트 실행 설명서

## 폴더 구조

```
ADV/
├── config/                # 설정 파일 및 환경설정
├── data/
│   ├── raw/               # 원본 데이터 (git에 업로드 X)
│   ├── processed/         # 전처리된 데이터 (git에 업로드 X)
│   ├── backup/            # 데이터 백업 (git에 업로드 X)
│   ├── metadata/          # 데이터 관련 메타정보 (git에 업로드 X)
│   ├── augmentation.py    # 데이터 증강 함수
│   ├── cache.py           # 데이터 캐싱 관련 코드
│   ├── dataset.py         # 데이터셋 클래스
│   ├── preprocessor.py    # 데이터 전처리 함수
│   └── splitter.py        # 데이터 분할 함수
├── eda_results/           # EDA 결과 (git에 업로드 X)
├── experiments/           # 실험 결과 (git에 업로드 X)
├── models/                # 모델 정의 및 저장
│   ├── backbone.py        # 백본 모델
│   ├── classifier.py      # 분류기
│   └── moco.py            # MoCo 관련 모델
├── scripts/               # 실행 스크립트
│   ├── run_eda.py         # EDA 실행 스크립트
│   ├── run_finetune.py    # 파인튜닝 스크립트
│   ├── run_pretrain.py    # 사전훈련 스크립트
│   ├── run_test.py        # 테스트 스크립트
│   └── experiments/       # 실험별 실행 스크립트
├── trainers/              # 학습 관련 코드
│   ├── finetune.py        # 파인튜닝 트레이너
│   ├── pretrain.py        # 사전훈련 트레이너
│   ├── test.py            # 테스트 모듈
├── utils/                 # 유틸리티 함수
│   ├── eda.py             # EDA 관련 함수
│   ├── logger.py          # 로깅 함수
│   └── metrics.py         # 평가 지표 함수
├── wandb/                 # W&B 로그 (git에 업로드 X)
├── wandb_logs/            # W&B 추가 로그 (git에 업로드 X)
├── Multi_label_Moco_0607.ipynb # 원본 Jupyter 노트북
├── requirements.txt       # 패키지 목록
├── .env                   # 패키지 및 환경설정 파일
├── .gitignore             # git 제외 파일 목록
```

## 주요 코드 설명

- **config/** : 하이퍼파라미터, 경로 등 설정 파일
- **data/** : 데이터 로딩, 전처리, 증강, 분할 등 데이터 관련 코드
- **models/** : 모델 구조 정의 (백본, 분류기, MoCo 등)
- **trainers/** : 사전훈련, 파인튜닝, 테스트 관련 코드
- **utils/** : EDA, 로깅, 평가 지표 등 보조 함수
- **scripts/** : EDA, 사전훈련, 파인튜닝, 테스트 스크립트
- **experiments/**, **eda_results/**, **wandb/**, **wandb_logs/** : 실험 결과 및 로그 (git에 업로드 X)
- **Multi_label_Moco_0607.ipynb** : 원본 Jupyter 노트북

## 실행 방법

### 1. 환경 설정

**가상환경 생성 및 활성화**
```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source .venv/bin/activate
```

**필수 패키지 설치**
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
`data/raw/` 폴더에 필요한 오디오 데이터(.wav 파일)와 메타데이터를 넣으세요.
(현재 폴더에 .wav 파일들이 모두 업로드되어 있습니다)

### 3. 실험 실행
아래는 예시 구문입니다. bash에 입력하여 각 단계별로 학습/평가를 수행합니다.

**STEP 1. Pretraining**
필수 인자: `--exp`: 실험 세팅값
```bash
PYTHONPATH=. python scripts/run_pretrain.py --exp exp001
```

**STEP 2. Fine-tuning**
필수 인자: `--exp`: 실험 세팅값, `--ssl-checkpoint`: 사전훈련 인코더 가중치 경로
```bash
PYTHONPATH=. python scripts/run_finetune.py --exp exp001 --ssl-checkpoint /checkpoints/exp001/best_pretrained_model.pth
```

**STEP 3. Test**
필수 인자: `--exp`: 실험 세팅값, `--ssl-checkpoint`: 사전훈련 인코더+분류기 가중치 경로
```bash
PYTHONPATH=. python scripts/run_test.py --exp exp001 --ssl-checkpoint /checkpoints/exp001/best_finetuned_model.pth
```

**EDA(Exploratory Data Analysis) 실행**
```bash
PYTHONPATH=. python scripts/run_eda.py
```

### 4. 실험 설정 변경
`scripts/experiments/` 폴더에서 실험 설정을 수정하거나 새로운 실험을 추가할 수 있습니다.
- `exp001.py`: 기본 SSL + Fine-tuning 실험
- `exp002.py`: 다양한 데이터 증강 실험

### 5. 실험 결과 확인
- **학습 로그**: `wandb/` 폴더 또는 W&B 웹 대시보드
- **체크포인트**: `checkpoints/exp001/` 폴더
- **EDA 결과**: `eda_results/` 폴더

---

추가로 궁금한 점이나, 세부 실행법이 필요한 부분이 있으면 README에 덧붙여 주세요!
