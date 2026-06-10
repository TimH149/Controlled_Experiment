import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------------------
# Settings
# ----------------------------

DATASET_DIR = Path("tennis_balloon_dataset")

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-3


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Dataset loading
# ----------------------------

def make_loader(split: str, shuffle: bool) -> DataLoader:
    """Load a split with torchvision's ImageFolder format"""
    split_path = DATASET_DIR / split

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root=str(split_path),
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


# ----------------------------
# Small CNN 
# ----------------------------

class SmallCNN(nn.Module):
    """A small CNN used to test whether the dataset creates a color shortcut"""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------
# Training and evaluation
# ----------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(train_split, train_loader, val_loader, num_classes):
    """Train fresh model on one training split"""
    model = SmallCNN(num_classes=num_classes).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining on {train_split}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            loss_fn,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.3f} | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.3f}"
        )

    return model, loss_fn


def test_model(model, loss_fn, test_loaders):
    """Evaluate trained model on all test splits"""
    results = {}

    print("\nFinal evaluation")
    print("----------------")

    for split_name, loader in test_loaders.items():
        _, accuracy = evaluate(model, loader, loss_fn)
        results[split_name] = accuracy
        print(f"{split_name:15s} accuracy: {accuracy:.3f}")

    return results


def interpret_results(results):
    """Print a short interpretation of the biased-to-reversed accuracy drop"""
    biased_acc = results["test_biased"]
    reversed_acc = results["test_reversed"]
    drop = biased_acc - reversed_acc

    print("\nInterpretation")
    print("--------------")
    print(f"Accuracy drop from biased to reversed test set: {drop:.3f}")

    if drop > 0.30:
        print(
            "The model likely relied strongly on the color shortcut. "
            "It performs well when the training color-label relation holds, "
            "but fails when the colors are reversed."
        )
    elif drop > 0.10:
        print(
            "The model may partially rely on the color shortcut. "
            "There is a noticeable performance drop under color reversal."
        )
    else:
        print(
            "The model does not appear strongly fooled by the color reversal. "
            "It likely learned object structure rather than color."
        )


def main():
    print(f"Using device: {DEVICE}")

    train_splits = {
        "train_biased": make_loader("train_biased", shuffle=True),
        "train_balanced": make_loader("train_balanced", shuffle=True),
        "train_colored": make_loader("train_colored", shuffle=True),
    }

    val_loader = make_loader("val_biased", shuffle=False)

    test_loaders = {
        "test_biased": make_loader("test_biased", shuffle=False),
        "test_balanced": make_loader("test_balanced", shuffle=False),
        "test_reversed": make_loader("test_reversed", shuffle=False),
    }

    class_names = train_splits["train_biased"].dataset.classes
    num_classes = len(class_names)

    print(f"Classes: {class_names}")

    all_results = {}

    for train_split, train_loader in train_splits.items():
        model, loss_fn = train_model(
            train_split=train_split,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
        )

        results = test_model(model, loss_fn, test_loaders)
        interpret_results(results)

        all_results[train_split] = results

        model_path = f"shortcut_cnn_{train_split}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved model to {model_path}")

    print("\nSummary")
    print("-------")
    print(f"{'Training split':15s} {'test_biased':>12s} {'test_balanced':>14s} {'test_reversed':>14s}")

    for train_split, results in all_results.items():
        print(
            f"{train_split:15s} "
            f"{results['test_biased']:12.3f} "
            f"{results['test_balanced']:14.3f} "
            f"{results['test_reversed']:14.3f}"
        )


if __name__ == "__main__":
    main()