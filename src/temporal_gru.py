import torch
import torch.nn as nn


class TemporalGRU(nn.Module):
    """GRU temporal utilizada sobre la secuencia de probabilidades visuales.

    Entrada por instante: [p_t, delta_p_t, delta_t].
    La salida es un logit de alerta para la ventana causal completa.
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        last = output[:, -1, :]
        return self.head(last).squeeze(-1)


def load_temporal_checkpoint(checkpoint_path: str, device: str | torch.device = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = TemporalGRU(
        input_size=len(config.get("features", ["feat_p", "feat_dp", "feat_dt"])),
        hidden_size=32,
        num_layers=1,
        dropout=0.2,
    )

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    threshold = 0.85
    if isinstance(checkpoint, dict):
        threshold = checkpoint.get("threshold", checkpoint.get("val_threshold", threshold))

    return model, float(threshold), config


def alert_probability(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.sigmoid(model(x))
