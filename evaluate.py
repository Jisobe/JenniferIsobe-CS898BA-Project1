from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from preprocess import build_dataloaders
from model import FishClassifier
from train import DEVICE

CURRENT_DIR = Path.cwd()
RESULTS_DIR = CURRENT_DIR / "hw3-results"
BASELINE_DIR = RESULTS_DIR / "baseline"
TEST_DIR = RESULTS_DIR / "test"
EVAL_DIR = RESULTS_DIR / "evaluation"
BASELINE_MODEL_PATH = BASELINE_DIR / "baseline_model.pt"
BASELINE_HISTORY_PATH = BASELINE_DIR / "baseline_history.json"
OPTIMIZED_MODEL_PATH = TEST_DIR / "best_model.pt"
OPTIMIZED_HISTORY_PATH = TEST_DIR / "best_history.json"
COMPARISON_GRID_PATH = EVAL_DIR / "comparison_grid.png"
BASELINE_REPORT_PATH = EVAL_DIR / "baseline_classification_report.txt"
OPTIMIZED_REPORT_PATH = EVAL_DIR / "optimized_classification_report.txt"

# Create the directories in the current directory if the do not exist
EVAL_DIR.mkdir(exist_ok=True)

def load_history(path):
    with open(path) as f:
        return json.load(f)

def get_predictions(model, loader):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_predictions)

def build_model_from_checkpoint(model_path, num_classes):
    model = FishClassifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model

def report_and_save(y_true, y_pred, class_names, save_path, model_label):
    report_str = classification_report(y_true, y_pred, target_names=class_names,digits=4, zero_division=0)
    print(f"\n{'=' * 60}\n{model_label} - Classification Report\n{'=' * 60}")
    print(report_str)

    with open(save_path, "w") as f:
        f.write(f"{model_label} - Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
    print(f"Saved to {save_path}")

    return report_str

def build_comparison_grid(baseline_hist, optimized_hist, y_true_opt, y_pred_opt,class_names):
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.3])

    ax1 = fig.add_subplot(gs[0, 0])
    epochs_b = range(1, len(baseline_hist["train_loss"]) + 1)
    ax1.plot(epochs_b, baseline_hist["train_loss"], label="Train Loss")
    ax1.plot(epochs_b, baseline_hist["validation_loss"], label="Validation Loss")
    ax1.set_title("Baseline: Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs_b, baseline_hist["train_accuracy"], label="Train Accuracy")
    ax2.plot(epochs_b, baseline_hist["validation_accuracy"], label="Validation Accuracy")
    ax2.set_title("Baseline: Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 1])
    epochs_o = range(1, len(optimized_hist["train_loss"]) + 1)
    ax3.plot(epochs_o, optimized_hist["train_loss"], label="Train Loss")
    ax3.plot(epochs_o, optimized_hist["validation_loss"], label="Validation Loss")
    ax3.set_title("Optimized: Loss")
    ax3.set_xlabel("Epoch")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs_o, optimized_hist["train_accuracy"], label="Train Accuracy")
    ax4.plot(epochs_o, optimized_hist["validation_accuracy"], label="Validation Accuracy")
    ax4.set_title("Optimized: Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.legend()

    ax5 = fig.add_subplot(gs[:, 2])
    cm = confusion_matrix(y_true_opt, y_pred_opt)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax5, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax5.set_title("Optimized Model: Test Confusion Matrix")

    plt.tight_layout()
    plt.savefig(COMPARISON_GRID_PATH, dpi=150)
    print(f"\nSaved comparison grid to {COMPARISON_GRID_PATH}")

def main():
    baseline_data = load_history(BASELINE_HISTORY_PATH)
    optimized_data = load_history(OPTIMIZED_HISTORY_PATH)

    class_to_index = baseline_data["class_to_index"]
    num_classes = len(class_to_index)
    class_names = [name for name, index in sorted(class_to_index.items(), key=lambda kv: kv[1])]

    _, _, test_loader, _ = build_dataloaders()

    baseline_model = build_model_from_checkpoint(BASELINE_MODEL_PATH, num_classes)
    optimized_model = build_model_from_checkpoint(OPTIMIZED_MODEL_PATH, num_classes)

    y_true_base, y_predict_base = get_predictions(baseline_model, test_loader)
    y_true_optimized, y_predict_optimized = get_predictions(optimized_model, test_loader)

    report_and_save(y_true_base, y_predict_base, class_names,
                     BASELINE_REPORT_PATH, "Baseline Model")
    report_and_save(y_true_optimized, y_predict_optimized, class_names,
                     OPTIMIZED_REPORT_PATH, "Optimized Model")

    print(f"\nBaseline - test_loss={baseline_data['test_loss']:.4f} "
          f"test_accuracy={baseline_data['test_accuracy']:.4f}")
    print(f"Optimized - test_loss={optimized_data['test_loss']:.4f} "
          f"test_accuracy={optimized_data['test_acc']:.4f} (config: {optimized_data['config']})")

    build_comparison_grid(baseline_data["history"], optimized_data["history"], y_true_optimized, y_predict_optimized, class_names)

if __name__ == "__main__":
    main()