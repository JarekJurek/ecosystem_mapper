import itertools
import subprocess
from pathlib import Path

BASE_CONFIG = "src/configs/default.yaml"

LR_LIST = [1e-4, 3e-4, 1e-3]
BATCH_SIZES = [32, 64]
DROPOUTS = [0.2, 0.5]
NUM_HIDDEN= [64, 128, 256]
BATCH_NORMS = [True, False]

def main():
    for lr, batch_size, dropout, var_hidden, use_batchnorm in itertools.product(LR_LIST, BATCH_SIZES, DROPOUTS, NUM_HIDDEN, BATCH_NORMS):
        exp_name = f"mlp_lr{lr}_bs{batch_size}_do{dropout}_vh{var_hidden}_bn{use_batchnorm}".replace(".", "p")

        print(f"\n=== Running {exp_name} ===\n")

        cmd = [
            "python3",
            Path("src/main.py"),
            "--lr", str(lr),
            "--batch_size", str(batch_size),
            "--dropout", str(dropout),
            "--out_dir", exp_name,
            "--var_hidden", str(var_hidden),
            "--use_batchnorm", str(use_batchnorm).lower()
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
