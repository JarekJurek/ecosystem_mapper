import itertools
import subprocess
from pathlib import Path

BASE_CONFIG = "src/configs/default.yaml"

LR_LIST = [0.003, 0.0005, 0.0008, 0.002, 0.005]
BATCH_SIZES = [16, 128, 256]
DROPOUTS = [0.0, 0.1, 0.15, 0.25, 0.3]
NUM_HIDDEN = [384, 512, 768, 1024]
WEIGTH_DECAY = [0, 1e-4, 5e-5, 1e-5]
BATCH_NORMS = [True]

def main():
    for lr, batch_size, dropout, var_hidden, use_batchnorm, weight_decay in itertools.product(LR_LIST, BATCH_SIZES, DROPOUTS, NUM_HIDDEN, BATCH_NORMS, WEIGTH_DECAY):
        exp_name = f"mlp_lr{lr}_bs{batch_size}_do{dropout}_vh{var_hidden}_bn{use_batchnorm}_wd{weight_decay}".replace(".", "p")

        print(f"\n=== Running {exp_name} ===\n")

        cmd = [
            "python3",
            Path("src/main.py"),
            "--lr", str(lr),
            "--batch_size", str(batch_size),
            "--dropout", str(dropout),
            "--out_dir", exp_name,
            "--var_hidden", str(var_hidden),
            "--use_batchnorm", str(use_batchnorm).lower(),
            "--weight_decay", str(weight_decay)
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
