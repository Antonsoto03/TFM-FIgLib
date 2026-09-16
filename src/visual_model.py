import torch
import torch.nn as nn
from torchvision import models


def build_full384_model(pretrained: bool = False) -> nn.Module:
    """Construye el clasificador visual ResNet18 usado como base Full384.

    La última capa se sustituye por una salida binaria (logit de humo).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_full384_checkpoint(checkpoint_path: str, device: str | torch.device = "cpu") -> nn.Module:
    model = build_full384_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def smoke_probability(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Devuelve P(humo|imagen) para un batch ya preprocesado."""
    with torch.no_grad():
        logits = model(batch).squeeze(-1)
        return torch.sigmoid(logits)
