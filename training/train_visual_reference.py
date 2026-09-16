"""Implementación de referencia del entrenamiento Full384.

Este script refleja la configuración documentada en el TFM (ResNet18, 384 px,
AdamW, LR 3e-4, weight decay 1e-4, cosine scheduler, AMP y acumulación de
gradiente). No reemplaza los notebooks históricos de Colab, pero permite
reproducir el núcleo del entrenamiento con un manifiesto equivalente.

Formato esperado del CSV:
path,label,event_id
/ruta/imagen1.jpg,0,evento_001
/ruta/imagen2.jpg,1,evento_002
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.visual_model import build_full384_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ManifestDataset(Dataset):
    def __init__(self, csv_path: str, train: bool):
        self.df = pd.read_csv(csv_path)
        if train:
            self.transform = transforms.Compose([
                transforms.Resize(440),
                transforms.RandomResizedCrop(384, scale=(0.65, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(440),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(Path(row["path"])).convert("RGB")
        x = self.transform(image)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(-1)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            n += len(y)
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--output", default="models/full384_reference.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_full384_model(pretrained=True).to(device)

    train_loader = DataLoader(ManifestDataset(args.train_csv, train=True), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(ManifestDataset(args.val_csv, train=False), batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x).squeeze(-1)
                loss = loss_fn(logits, y) / args.accumulation
            scaler.scale(loss).backward()

            if step % args.accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch:03d} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "input": 384,
                    "backbone": "resnet18",
                    "lr": 3e-4,
                    "weight_decay": 1e-4,
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.accumulation,
                },
            }, args.output)


if __name__ == "__main__":
    main()
