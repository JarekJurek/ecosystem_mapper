import torch
import torch.nn as nn
from tqdm.auto import tqdm

from dataset.dataset import get_dataloaders
from model import FusionNet


def train(model, train_loader, val_loader, optimizer, device, loss_fn, epochs: int):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))
        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}"
        )


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
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
            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))
    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def main():
    # Simple scalar config variables
    csv_path = "data/dataset_split.csv"
    image_dir = None
    image_ext = ".png"
    variable_selection = "all"  # can be list, string, or None
    batch_size = 32
    num_workers = 4
    epochs = 5
    lr = 1e-3
    resnet_name = "resnet18"  # "resnet18" or "resnet50"
    pretrained = True
    var_hidden = 128
    dropout = 0.2
    num_classes = 17
    load_images = True  # set False for variables-only mode

    loaders = get_dataloaders(
        csv_path=csv_path,
        image_dir=image_dir,
        image_ext=image_ext,
        variable_selection=variable_selection,
        batch_size=batch_size,
        num_workers=num_workers,
        load_images=load_images,
    )

    sample_batch = next(iter(loaders["train"]))
    var_dim = sample_batch.get("variables")
    var_input_dim = var_dim.shape[1] if var_dim is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNet(
        num_classes=num_classes,
        var_input_dim=var_input_dim,
        var_hidden_dim=var_hidden,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train(model, loaders["train"], loaders["val"], optimizer, device, loss_fn, epochs)

    test_loss, test_acc = evaluate(model, loaders["test"], device, loss_fn)
    print(f"Test | loss {test_loss:.4f} acc {test_acc:.3f}")


if __name__ == "__main__":
    main()
