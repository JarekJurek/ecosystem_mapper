import matplotlib.pyplot as plt
import torchvision.utils as vutils
from config import config as cfg
import torch
import os

def show_images_from_loader(data_loader, num_images=5):
    """
    Display images from a given DataLoader.

    Args:
    - data_loader: The DataLoader to fetch images from.
    - num_images: The number of images to display.
    """
    for i, batch in enumerate(data_loader):
        images = batch['images']
        if images is None:
            print("No images found in this batch.")
            continue
        labels = batch['label_names']

        plt.figure(figsize=(15, 5))

        for j in range(min(num_images, images.size(0))):
            plt.subplot(1, num_images, j + 1)
            plt.imshow(vutils.make_grid(images[j:j+1], nrow=1).permute(1, 2, 0).numpy())
            plt.title(f'Label: {labels[j]}')
            plt.axis('off')

        plt.show()
        if i >= 0:
            break

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Using Apple MPS GPU (Metal Performance Shaders)")
        return torch.device("mps")

    print("Using CPU (no GPU backend available)")
    return torch.device("cpu")

def save_checkpoint(model, optimizer, epoch, name="checkpoint.pth", extra=None):
    """Save model/optimizer state safely."""
    if not cfg.save_checkpoint:
        return

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, os.path.join(cfg.checkpoints_dir, name))
    print(f"[CHECKPOINT] Saved checkpoint {name}")

def load_checkpoint(model, optimizer, ckpt_path, device):
    """
    Load a checkpoint if it exists.

    Parameters
    ----------
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    ckpt_path : str
        Full path to checkpoint file.
    device : str or torch.device
    """
    if not cfg.load_checkpoint:
        return
    if not os.path.exists(ckpt_path):
        print(f"[CHECKPOINT] No checkpoint found at {ckpt_path}.")
        return

    print(f"[CHECKPOINT] Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt.get("model_state", {}))

    if "optimizer_state" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print("[CHECKPOINT] Loaded model & optimizer.")
    else:
        print("[CHECKPOINT] Loaded model only (no optimizer state).")
