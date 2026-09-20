import torch

from src.temporal_gru import TemporalGRU
from src.visual_model import build_full384_model


def test_visual_model_output_shape():
    model = build_full384_model(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1)


def test_temporal_gru_output_shape():
    model = TemporalGRU(input_size=3, hidden_size=32, num_layers=1, dropout=0.2)
    model.eval()
    x = torch.randn(2, 8, 3)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2,)
