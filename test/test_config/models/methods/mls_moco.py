class MLSMocoConfig:
    def __init__(
        self,
        K = 512,             # memory queue size
        m = 0.999,           # momentum
        T = 0.07,            # temperature
        dim_mlp = 128,       # projector q,k의 output z1,z2의 차원
        lambda_bce = 0.5,    # BCE loss에 곱해지는 lambda
        top_k = 10,          # positive pair의 개수
        warmup_epochs = 10   # 초기에 InfoNCE loss만 사용하는 epoch, default: 10
    ):
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.dim_mlp = dim_mlp
        self.lambda_bce = lambda_bce
        self.top_k = top_k
        self.warmup_epochs = warmup_epochs