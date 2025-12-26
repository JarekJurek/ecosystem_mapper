import time
from pathlib import Path
from dataset.dataset import get_dataloaders
import torchvision.transforms as T

def benchmark_loader(batch_size, num_workers, n_batches=30, load_images=False):
    data_dir = Path(__file__).parents[1].resolve() / "data"
    csv_path = data_dir / "dataset_split.csv"
    image_dir = data_dir / "preprocessed_png_256"

    eff_mean = [0.485, 0.456, 0.406]
    eff_std  = [0.229, 0.224, 0.225]
    img_size = 224

    train_transform = T.Compose([
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
    ])

    eval_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=eff_mean, std=eff_std),
    ])

    loaders = get_dataloaders(
        csv_path=csv_path,
        image_dir=image_dir,
        image_ext=".png",
        variable_selection="all",
        batch_size=batch_size,
        num_workers=num_workers,
        train_image_transform=train_transform,
        eval_image_transform=eval_transform,
        load_images=load_images,
    )

    train_loader = loaders["train"]

    start = time.time()
    total_samples = 0
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        bs = batch["labels"].shape[0] if batch["labels"] is not None else 0
        total_samples += bs
    elapsed = time.time() - start
    if elapsed == 0:
        return

    print(
        f"bs={batch_size}, workers={num_workers} "
        f"| {elapsed:.2f}s for {n_batches} batches "
        f"| {total_samples/elapsed:.1f} samples/s"
    )

if __name__ == "__main__":
    for bs in [512, 1024, 2048]:
        for nw in [12, 16, 18, 20]:
            benchmark_loader(bs, nw)
