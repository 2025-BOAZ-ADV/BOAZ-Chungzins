class ClassifierConfig:
    def __init__(
        self,
        in_dim = 2048,
        layer_dims = [512, 128],
        dropout_rate = 0.5,
        freeze_encoder = True
    ):
        super().__init__()

        self.in_dim = in_dim
        self.layer_dims = layer_dims
        self.dropout_rate = dropout_rate
        self.freeze_encoder = freeze_encoder

        layers_info = f"Linear({in_dim} -> "
        for out_dim in layer_dims:
            layers_info += f"{out_dim} -> "
        layers_info += f"2)"

        self.num_layers = len(layer_dims) + 1
        self.classifier_architecture = layers_info
