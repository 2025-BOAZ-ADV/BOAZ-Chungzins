class FinetuneConfig:
    def __init__(
        self,
        batch_size = 64,
        num_workers = 4,
        epochs = 100,
        learning_rate = 0.03,
        weight_decay = 0.01,
        use_cache = False,    # 오디오를 mel spectrogram으로 변환하는 작업을 건너뛰고 캐시를 불러올지 설정
        save_cache = True     # 캐시 저장 여부 (오디오 파일명을 해시로 변환한 것이므로, 새로 저장 시 덮어쓰기가 됨)
    ):
        # 훈련 파라미터
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # 캐시 사용 여부 (캐시 = ICBHI train data의 각 cycle의 mel spectrogram을 다음에 불러오기 쉽게 .pt 파일로 백업해놓은 것)
        self.use_cache = use_cache
        self.save_cache = save_cache