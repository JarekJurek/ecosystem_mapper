import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn

from dataset.dataset import get_dataloaders
from model import FusionNet
from metrics_plots import (
    plot_training_curves,
    compute_confusion_matrix,
    plot_confusion_matrix,
    ensure_dir,
)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    loss_fn,
    epochs: int,
    scheduler=None,
    patience: int = 0,
    grad_clip: float = 1.0,
):
    """Train loop returning metrics dict for plotting."""
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    expected_train = len(getattr(train_loader, "dataset", []))
    expected_val = len(getattr(val_loader, "dataset", []))
    print(f"Dataset sizes | train: {expected_train} | val: {expected_val}")
        
    best_val_loss = float("inf")
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in train_loader:
            images = batch.get("images")
            variables = batch.get("variables")
            labels = batch.get("labels")
            if labels is None:
                continue
            if images is not None:
                images = images.to(device)
            if variables is not None:
                variables = variables.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images, variables)
            loss = loss_fn(logits, labels)
            loss.backward()

            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(batch_size)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step()

        # Early stopping based on validation loss
        if patience and val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improve = 0
        elif patience:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val loss improve for {patience} epochs)"
                )
                break
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}"
        )
    return metrics


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            images = batch.get("images")
            variables = batch.get("variables")
            labels = batch.get("labels")
            if labels is None:
                continue
            if images is not None:
                images = images.to(device)
            if variables is not None:
                variables = variables.to(device)
            labels = labels.to(device)
            logits = model(images, variables)
            loss = loss_fn(logits, labels)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(batch_size)
    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train Fusion model")
    parser.add_argument(
        "--variable-selection",
        dest="variable_selection",
        nargs="*",
        default=["all"],
        help="Variable selection: 'all', a group key, or multiple keys (e.g., geol edaph)",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parents[1].resolve() / "data"
    csv_path = data_dir / "dataset_split.csv"
    image_dir = data_dir / "images"
    variable_selection = args.variable_selection

    ### Hyperparams ###
    batch_size = 32
    num_workers = 6
    epochs = 30
    lr = 1e-3
    weight_decay = 5e-4
    label_smoothing = 0.10
    early_stopping_patience = 10
    var_hidden = 256
    dropout = 0.3
    num_classes = 17
    load_images = True
    grad_clip = 1.0

    loaders = get_dataloaders(
        csv_path=csv_path,
        image_dir=image_dir,
        variable_selection=variable_selection,
        batch_size=batch_size,
        num_workers=num_workers,
        load_images=load_images,
    )

    sample_batch = next(iter(loaders["train"]))
    var_tensor = sample_batch.get("variables")
    var_input_dim = var_tensor.shape[1] if var_tensor is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNet(
        num_classes=num_classes,
        var_input_dim=var_input_dim,
        var_hidden_dim=var_hidden,
        dropout=dropout,
    ).to(device)

    ### Backbone partial fine-tuning ###
    for name, p in model.backbone.named_parameters():
        if "features.6" in name or "features.7" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    backbone_params = [
        p
        for n, p in model.named_parameters()
        if n.startswith("backbone") and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("backbone")
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    metrics = train(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        device,
        loss_fn,
        epochs,
        scheduler=scheduler,
        patience=early_stopping_patience,
        grad_clip=grad_clip,
    )

    test_loss, test_acc = evaluate(model, loaders["test"], device, loss_fn)
    print(f"Test | loss {test_loss:.4f} acc {test_acc:.3f}")

    # Plotting and confusion matrix generation
    out_dir = "models"
    ensure_dir(out_dir)
    curves_path = os.path.join(out_dir, "training_curves.png")
    plot_training_curves(metrics, curves_path)
    print(f"Saved training curves to {curves_path}")

    cm_tensor, class_names = compute_confusion_matrix(model, loaders["val"], device)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm_tensor, class_names, cm_path)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
