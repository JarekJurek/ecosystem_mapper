from metrics_plots import (
    plot_training_curves,
    compute_confusion_matrix,
    plot_confusion_matrix,
)
from model import FusionNet
import os
import torch.nn as nn
import torchvision.transforms as T
from dataset.dataset import get_dataloaders
from utils import get_best_device, save_checkpoint
from config import config as cfg
from train_fusion import train
import torch
from train_fusion import evaluate

def main():
    # https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
    eff_mean=[0.485, 0.456, 0.406]
    eff_std = [0.229, 0.224, 0.225]
    img_size = 220

    eval_transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=eff_mean, std=eff_std),
        ]
    )

   # Trainings augmentations
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=90),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            T.ToTensor(),
            T.Normalize(mean=eff_mean, std=eff_std),
        ]
    )


    loaders = get_dataloaders(
        image_ext=".png",
        csv_path=cfg.csv_path,
        image_dir=cfg.image_dir,
        variable_selection=cfg.variable_selection,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        load_images=cfg.load_images,
        train_image_transform=train_transform,
        eval_image_transform=eval_transform,
    )

    sample_batch = next(iter(loaders["train"]))

    var_tensor = sample_batch.get("variables")
    var_input_dim = var_tensor.shape[1] if var_tensor is not None else None
    print(f"Variables used for this run: {cfg.variable_selection}")

    device = get_best_device()
    model = FusionNet(
        num_classes=cfg.num_classes,
        var_input_dim=var_input_dim,
        var_hidden_dim=cfg.var_hidden,
        dropout=cfg.dropout,
    ).to(device)

    ### Backbone partial fine-tuning ###
    for name, p in model.backbone.named_parameters():
        if "features.5" in name or "features.6" in name or "features.7" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    backbone_params = [
        p for n, p in model.named_parameters()
        if n.startswith("backbone") and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("backbone")
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr * 0.05},
            {"params": head_params, "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    # load checkpoint
    ckpt_path = f"{cfg.checkpoints_dir}/best_checkpoint.pth"
    if os.path.exists(ckpt_path):
        print(f"[CHECKPOINT] Found {ckpt_path}, loading...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[CHECKPOINT] Loaded model & optimizer from {ckpt_path}")
    else:
        print("[CHECKPOINT] No existing checkpoint found, training from scratch.")


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    try:
        metrics = train(
            model,
            loaders["train"],
            loaders["val"],
            optimizer,
            device,
            loss_fn,
            cfg.epochs,
            scheduler=scheduler,
            patience=cfg.early_stopping_patience,
            grad_clip=cfg.grad_clip,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user. Saving checkpoint...")
        save_checkpoint(model, optimizer, epoch="interrupted", name="checkpoint_interrupt.pth")
        return


    test_loss, test_acc = evaluate(model, loaders["test"], device, loss_fn)
    print(f"Test | loss {test_loss:.4f} acc {test_acc:.3f}")

    curves_path = os.path.join(cfg.results_dir, "training_curves.png")
    plot_training_curves(metrics, curves_path)
    print(f"Saved training curves to {curves_path}")

    cm_tensor, class_names = compute_confusion_matrix(model, loaders["val"], device)
    cm_path = os.path.join(cfg.results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm_tensor, class_names, cm_path)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
