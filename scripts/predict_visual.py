import argparse
from pathlib import Path

import torch
from PIL import Image

from src.preprocessing import full384_eval_transform
from src.visual_model import load_full384_checkpoint, smoke_probability


def main():
    parser = argparse.ArgumentParser(description="Inferencia Full384 sobre una imagen FIgLib/HPWREN")
    parser.add_argument("--image", required=True, help="Ruta a la imagen")
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint visual")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral para decisión binaria")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_full384_checkpoint(args.checkpoint, device=device)

    image = Image.open(Path(args.image)).convert("RGB")
    x = full384_eval_transform()(image).unsqueeze(0).to(device)
    p = float(smoke_probability(model, x).item())

    print(f"p_smoke={p:.6f}")
    print(f"prediction={'smoke' if p >= args.threshold else 'no_smoke'}")


if __name__ == "__main__":
    main()
