from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix


def plot_training_curves(history: Dict[str, List[float]], out_path: str) -> None:
    """Plot training and validation loss/accuracy curves.

    Expects history keys: train_loss, val_loss, train_acc, val_acc.
    Saves a PNG to out_path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_confusion_matrix(model: torch.nn.Module, loader, device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    """Compute confusion matrix on a loader. Returns (cm_tensor, class_names list)."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    id_to_name: Dict[int, str] = {}
    with torch.no_grad():
        for batch in loader:
            images = batch.get("images")
            variables = batch.get("variables")
            labels = batch.get("labels")
            label_names = batch.get("label_names")
            if labels is None:
                continue
            if images is not None:
                images = images.to(device)
            if variables is not None:
                variables = variables.to(device)
            labels = labels.to(device)
            logits = model(images, variables)
            preds = logits.argmax(dim=1)
            for p, l, name in zip(preds.cpu().tolist(), labels.cpu().tolist(), label_names):
                all_preds.append(p)
                all_labels.append(l)
                if l not in id_to_name and name is not None:
                    id_to_name[l] = str(name)
    sorted_ids = sorted(id_to_name.keys())
    class_names = [id_to_name[i] for i in sorted_ids]
    cm = confusion_matrix(all_labels, all_preds, labels=sorted_ids)
    return torch.tensor(cm, dtype=torch.int64), class_names


def plot_confusion_matrix(cm: torch.Tensor, class_names: List[str], out_path: str) -> None:
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(0.6 * len(class_names) + 4, 0.6 * len(class_names) + 4))
    plt.imshow(cm.numpy(), interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j].item())
            plt.text(j, i, str(val), ha="center", va="center", color="black", fontsize=8)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
