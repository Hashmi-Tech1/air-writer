"""Train a small CNN on EMNIST (balanced split) for air-written character recognition.

Usage:
    python train_model.py --epochs 8

This also defines the model class and label set that air_writer.py imports.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# EMNIST "balanced" has 47 classes (some lowercase letters are merged with uppercase).
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
MODEL_PATH = Path("emnist_cnn.pt")


class CharCNN(nn.Module):
    """Compact CNN: 2 x (conv-conv-pool) blocks followed by a small classifier."""

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _fix_orientation(x: torch.Tensor) -> torch.Tensor:
    # torchvision's EMNIST images are stored transposed; swap H and W to make them upright.
    return x.transpose(1, 2)


def get_loaders(batch_size: int, data_dir: str):
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Lambda(_fix_orientation),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(_fix_orientation)])
    train = datasets.EMNIST(data_dir, split="balanced", train=True, download=True, transform=train_tf)
    test = datasets.EMNIST(data_dir, split="balanced", train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=512, shuffle=False, num_workers=2),
    )


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the EMNIST character CNN")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default=str(MODEL_PATH))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_loader, test_loader = get_loaders(args.batch_size, args.data_dir)
    model = CharCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}/{args.epochs}  loss {running / len(train_loader):.4f}  test acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.out)

    print(f"Best test accuracy: {best:.4f}  (saved to {args.out})")


if __name__ == "__main__":
    main()