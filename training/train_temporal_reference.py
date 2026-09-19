"""Implementación de referencia del entrenamiento temporal GRU.

El CSV debe contener secuencias con las características ya preparadas:
event_id,order,feat_p,feat_dp,feat_dt,label

La normalización exacta de feat_dt debe ser la misma que en el experimento
original; el checkpoint no serializa ese factor.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.temporal_gru import TemporalGRU


class TemporalDataset(Dataset):
    def __init__(self, csv_path: str, seq_len: int = 8):
        df = pd.read_csv(csv_path)
        self.samples = []
        self.seq_len = seq_len

        for _, group in df.groupby("event_id"):
            group = group.sort_values("order").reset_index(drop=True)
            feats = group[["feat_p", "feat_dp", "feat_dt"]].to_numpy(dtype="float32")
            labels = group["label"].to_numpy(dtype="float32")

            for i in range(len(group)):
                if np.isnan(labels[i]):
                    continue
                start = max(0, i - seq_len + 1)
                w = feats[start : i + 1]
                if len(w) < seq_len:
                    pad = np.repeat(w[:1], seq_len - len(w), axis=0)
                    w = np.vstack([pad, w])
                self.samples.append((w, labels[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            ps.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    return roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de referencia de la GRU temporal")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--output", default="models/temporal_reference.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TemporalDataset(args.train_csv, seq_len=8)
    val_ds = TemporalDataset(args.val_csv, seq_len=8)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TemporalGRU(input_size=3, hidden_size=32, num_layers=1, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))

    best_auc = -1.0
    patience = 8
    bad_epochs = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        val_auc = evaluate(model, val_loader, device)
        print(f"epoch={epoch:03d} val_auc={val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "config": {"seq_len": 8, "features": ["feat_p", "feat_dp", "feat_dt"]},
            }, args.output)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break


if __name__ == "__main__":
    main()
