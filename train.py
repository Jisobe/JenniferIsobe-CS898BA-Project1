from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from preprocess import build_dataloaders
from model import FishClassifier

CURRENT_DIR = Path.cwd()
RESULTS_DIR = CURRENT_DIR / "hw3-results"
BASELINE_DIR = RESULTS_DIR / "baseline"
MODEL_SAVE_PATH = BASELINE_DIR / "baseline_model.pt"
CURVES_SAVE_PATH = BASELINE_DIR / "baseline_training_curves.png"
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the directories in the current directory if the do not exist
RESULTS_DIR.mkdir(exist_ok=True)
BASELINE_DIR.mkdir(exist_ok=True)

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            predications = torch.argmax(logits, dim=1)
            correct += (predications == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_baseline():
    print(f"Using device: {DEVICE}")

    train_loader, validation_loader, test_loader, class_to_index = build_dataloaders()
    num_classes = len(class_to_index)
    print(f"Classes ({num_classes}): {class_to_index}")

    model = FishClassifier(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [], "train_accuracy": [],
        "validation_loss": [], "validation_accuracy": [],
    }

    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} | "
              f"validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.4f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> new best validation_loss, saved weights to {MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)
    print(f"\nBaseline test set -> loss={test_loss:.4f} accuracy={test_acc:.4f}")

    plot_curves(history)

    return model, history, class_to_index

def plot_curves(history):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs_range, history["validation_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Baseline: Loss vs. Epoch")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(epochs_range, history["validation_accuracy"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Baseline: Accuracy vs. Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(CURVES_SAVE_PATH, dpi=150)
    print(f"Saved training curves to {CURVES_SAVE_PATH}")

if __name__ == "__main__":
    train_baseline()