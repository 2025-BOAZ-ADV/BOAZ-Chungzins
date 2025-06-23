class ClassifierConfig:
    def __init__(
        self,
        in_dim = 2048,
        layer_dims = [512, 128],
        dropout_rate = 0.5,
        freeze_encoder = True
    ):
        self.layer_dims = layer_dims
        self.dropout_rate = dropout_rate
        self.freeze_encoder = freeze_encoder

        import torch.nn as nn

        layers = []
        layers_info = f"Linear({in_dim} → "

        for out_dim in layer_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ]
            in_dim = out_dim
            layers_info += f"{out_dim} → "
        
        # 마지막 분류
        layers.append(nn.Linear(in_dim, 2))
        layers_info += f"2"

        self.num_layers = len(layer_dims) + 1
        self.classifier = nn.Sequential(*layers)
        self.classifier_architecture = layers_info
