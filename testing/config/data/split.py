class DataSplitConfig:
    def __init__(
        self,
        split_ratio = 0.75,
        allow_val = False,
        val_ratio = 0.25
    ):
        # train data를 사전훈련 set, 파인튜닝 set으로 분할할 때, 사전훈련 set의 비율
        self.split_ratio = split_ratio

        # 파인튜닝 set을 다시 train set, valid set으로 분할할 때, valid set의 비율
        self.allow_val = allow_val
        self.val_ratio = val_ratio