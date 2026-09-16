import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.temporal_gru import alert_probability, load_temporal_checkpoint


def build_features(df: pd.DataFrame, dt_scale: float | None = None) -> np.ndarray:
    if "p_smoke" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'p_smoke'.")

    p = df["p_smoke"].astype(float).to_numpy()
    dp = np.diff(p, prepend=p[0])

    if "feat_dt" in df.columns:
        dt = df["feat_dt"].astype(float).to_numpy()
    else:
        if "timestamp" not in df.columns:
            raise ValueError("Incluye 'feat_dt' o bien 'timestamp' junto con --dt-scale.")
        if dt_scale is None:
            raise ValueError(
                "El checkpoint no guarda el factor exacto de normalización temporal. "
                "Usa una columna 'feat_dt' ya normalizada o indica --dt-scale con el valor del experimento original."
            )
        t = df["timestamp"].astype(float).to_numpy()
        raw_dt = np.diff(t, prepend=t[0])
        dt = raw_dt / float(dt_scale)

    return np.column_stack([p, dp, dt]).astype("float32")


def causal_windows(features: np.ndarray, seq_len: int = 8) -> np.ndarray:
    windows = []
    for i in range(len(features)):
        start = max(0, i - seq_len + 1)
        w = features[start : i + 1]
        if len(w) < seq_len:
            pad = np.repeat(w[:1], seq_len - len(w), axis=0)
            w = np.vstack([pad, w])
        windows.append(w)
    return np.stack(windows)


def main():
    parser = argparse.ArgumentParser(description="Inferencia temporal con la GRU del TFM")
    parser.add_argument("--csv", required=True, help="CSV con p_smoke y feat_dt, o timestamp")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint best_temporal_gru_full384_38.pt")
    parser.add_argument("--threshold", type=float, default=0.85, help="Umbral operativo (validation)")
    parser.add_argument("--dt-scale", type=float, default=None, help="Factor de normalización de delta_t si no existe feat_dt")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    features = build_features(df, dt_scale=args.dt_scale)
    windows = causal_windows(features, seq_len=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, config = load_temporal_checkpoint(args.checkpoint, device=device)
    x = torch.from_numpy(windows).to(device)
    probs = alert_probability(model, x).cpu().numpy()

    out = df.copy()
    out["p_alert_gru"] = probs
    out["alert_gru"] = probs >= args.threshold

    print(f"checkpoint_config={config}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
