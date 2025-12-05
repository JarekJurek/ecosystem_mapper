import sys
from typing import Dict, Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils import save_checkpoint


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    epochs: int,
    scheduler: Optional[Any] = None,
    patience: int = 0,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, list]:
    """Train loop returning metrics dict for plotting.

    Works on both CPU and GPU, with optional AMP + grad clipping + early stopping.
    """
    model.to(device)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    expected_train = len(getattr(train_loader, "dataset", []))
    expected_val = len(getattr(val_loader, "dataset", []))
    print(f"Dataset sizes | train: {expected_train} | val: {expected_val}")

    best_val_loss = float("inf")
    epochs_without_improve = 0

    disable_tqdm = not sys.stdout.isatty()

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if disable_tqdm:
            batch_iter = train_loader
        else:
            batch_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs}",
                leave=False,
            )

        for step, batch in enumerate(batch_iter, start=1):
            images = batch.get("images")
            variables = batch.get("variables")
            labels = batch.get("labels")

            if labels is None:
                continue

            if images is not None:
                images = images.to(device, non_blocking=True)
            if variables is not None:
                variables = variables.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward + loss (with AMP on CUDA)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, variables)
                loss = loss_fn(logits, labels)

            # Backward + step with proper AMP handling
            if use_amp:
                scaler.scale(loss).backward()

                if grad_clip is not None and grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # --- Metrics accumulation (always, regardless of tqdm) ---
            batch_size = labels.size(0)
            total_samples += int(batch_size)
            total_loss += float(loss.item()) * batch_size

            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())

            if not disable_tqdm:
                current_loss = total_loss / max(total_samples, 1)
                current_acc = total_correct / max(total_samples, 1)
                batch_iter.set_postfix(
                    loss=f"{current_loss:.4f}",
                    acc=f"{current_acc:.3f}",
                )
        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # --- Validation ---
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        if scheduler is not None:
            if hasattr(scheduler, "step") and "metrics" in scheduler.step.__code__.co_varnames:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if patience:
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_without_improve = 0
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    name="best_checkpoint.pth",
                    extra={"val_loss": val_loss, "val_acc": val_acc},
                )
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no val loss improve for {patience} epochs)"
                    )
                    break

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
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
