class PreprocessConfig:
    def __init__(
        self,
        target_sec = 8,
        target_sr = 4000,
        frame_size = 1024,
        hop_length = 512,
        n_mels = 128
    ):
        super().__init__()
        
        self.target_sec = target_sec
        self.target_sr = target_sr
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.n_mels = n_mels