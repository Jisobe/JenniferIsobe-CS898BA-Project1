import json
from pathlib import Path
import csv
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import build_dataloaders, build_loaders_from_datasets, build_datasets
from model import FishClassifier
from train import run_epoch, DEVICE
import matplotlib.pyplot as plt

CURRENT_DIR = Path.cwd()
RESULTS_DIR = CURRENT_DIR / "hw3-results"
TEST_DIR = RESULTS_DIR / "test"
RESULTS_CSV = TEST_DIR / "test_results.csv"
BEST_MODEL_PATH = TEST_DIR / "best_model.pt"
BEST_CURVES_PATH = TEST_DIR / "best_model_training_curves.png"
BEST_HISTORY_PATH = TEST_DIR / "best_history.json"
LEARNING_RATES = [0.01, 0.001, 0.0001]
DROPOUT_RATES = [0.3, 0.5]
BATCH_SIZES = [32, 64]
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3]
TUNE_EPOCHS = 20
FINAL_EPOCHS = 30
PATIENCE = 7

# Create the directories in the current directory if the do not exist
RESULTS_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

def train_one_config(train_dataset, validation_dataset, test_dataset, class_to_index, learning_rate, batch_size, weight_decay, dropout, num_epochs):
    train_loader, validation_loader, test_loader = build_loaders_from_datasets(
        train_dataset, validation_dataset, test_dataset, batch_size=batch_size
    )
    num_classes = len(class_to_index)

    model = FishClassifier(num_classes=num_classes, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy

    return best_validation_loss, best_validation_accuracy, history

def run_grid_search():
    combos = list(itertools.product(LEARNING_RATES, BATCH_SIZES, WEIGHT_DECAYS, DROPOUT_RATES))
    print(f"Running grid search over {len(combos)} configurations with ({TUNE_EPOCHS} epochs each)\n")

    train_dataset, val_dataset, test_dataset, class_to_index = build_datasets()

    results = []
    best_overall = {"validation_loss": float("inf")}

    for i, (learning_rate, batch_size, weight_decay, dropout) in enumerate(combos, start=1):
        print(f"[{i}/{len(combos)}] learning_rate={learning_rate} batch_size={batch_size} weight_decay={weight_decay} dropout={dropout}")

        validation_loss, validation_accuracy, history = train_one_config(
            train_dataset, val_dataset, test_dataset, class_to_index,
            learning_rate, batch_size, weight_decay, dropout, num_epochs=TUNE_EPOCHS
        )

        print(f"  -> best_validation_loss={validation_loss:.4f} best_validation_accuracy={validation_accuracy:.4f}\n")

        result = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "best_validation_loss": validation_loss,
            "best_validation_accuracy": validation_accuracy,
        }
        results.append(result)

        if validation_loss < best_overall["validation_loss"]:
            best_overall = {**result, "validation_loss": validation_loss}

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved full grid search results to {RESULTS_CSV}")

    print(f"\nBest configuration:")
    print(f"  learning_rate = {best_overall['learning_rate']}")
    print(f"  batch_size = {best_overall['batch_size']}")
    print(f"  weight_decay = {best_overall['weight_decay']}")
    print(f"  validation_loss = {best_overall['best_validation_loss']:.4f}")
    print(f"  validation_accuracy = {best_overall['best_validation_accuracy']:.4f}")

    return best_overall, results

def retrain_best_config(best_overall):
    learning_rate = best_overall["learning_rate"]
    batch_size = best_overall["batch_size"]
    weight_decay = best_overall["weight_decay"]
    dropout = best_overall["dropout"]

    print(f"\nRetraining best config for {FINAL_EPOCHS} epochs "
          f"(learning_rate={learning_rate}, batch_size={batch_size}, weight_decay={weight_decay})")

    train_loader, validation_loader, test_loader, class_to_index= build_dataloaders(
        batch_size=batch_size
    )
    num_classes = len(class_to_index)

    model = FishClassifier(num_classes=num_classes, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epochs_without_improvement = 0
    best_validation_loss = float("inf")
    history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}

    for epoch in range(1, FINAL_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        print(f"Epoch {epoch:2d}/{FINAL_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} | "
              f"validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.4f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> new best validation_loss, saved weights to {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping: val_loss hasn't improved in {PATIENCE} epochs (stopped at epoch {epoch}).")
                break

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, optimizer=None)
    print(f"\nOptimized model test set -> loss={test_loss:.4f} accuracy={test_accuracy:.4f}")

    plot_curves_named(history, BEST_CURVES_PATH)

    with open(BEST_HISTORY_PATH, "w") as f:
        json.dump({
            "history": history,
            "class_to_idx": class_to_index,
            "test_loss": test_loss,
            "test_acc": test_accuracy,
            "config": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "dropout": dropout,
            },
        }, f, indent=2)
    print(f"Saved optimized model history to {BEST_HISTORY_PATH}")

    return model, history, class_to_index

def plot_curves_named(history, save_path):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs_range, history["validation_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Optimized Model: Loss vs. Epoch")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_accuracy"], label="Train accuracy")
    axes[1].plot(epochs_range, history["validation_accuracy"], label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Optimized Model: accuracy vs. Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved optimized model training curves to {save_path}")

if __name__ == "__main__":
    best_overall, all_results = run_grid_search()
    retrain_best_config(best_overall)