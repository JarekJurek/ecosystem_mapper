
def save_checkpoint(model, optimizer, epoch, path="./checkpoints/checkpoint.pth", extra=None):
    """Save model/optimizer state safely."""
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    print(f"[CHECKPOINT] Saved checkpoint to {path}")
