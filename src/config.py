from pathlib import Path
import configargparse
import os

def load_config():
    parser = configargparse.ArgParser(
        description="Train Fusion model",
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser.add(
        "-c", "--config",
        is_config_file=True,
        default="src/configs/default.yaml",
        help="Path to config file (YAML/INI).",
    )

    parser.add("--data_dir", type=str, default="data")
    parser.add("--csv_path", type=str, default=None)
    parser.add("--image_dir", type=str, default=None)
    parser.add("--load_checkpoint",type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add("--save_checkpoint",type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add("--use_batchnorm",type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add(
        "--variable_selection",
        action="append",
        default=[],
        help="Variables to select",
    )
    parser.add("--out_dir", default="new_exp")
    parser.add("--batch_size", type=int, default=32)
    parser.add("--num_workers", type=int, default=1)
    parser.add("--epochs", type=int, default=20)
    parser.add("--lr", type=float, default=1e-3)
    parser.add("--weight_decay", type=float, default=5e-4)
    parser.add("--label_smoothing", type=float, default=0.10)
    parser.add("--early_stopping_patience", type=int, default=10)
    parser.add("--var_hidden", type=int, default=256)
    parser.add("--dropout", type=float, default=0.3)
    parser.add("--num_classes", type=int, default=17)
    parser.add(
        "--load_images",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    parser.add("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()

    if args.config is not None:
        config_dir = Path(args.config).resolve().parent
    else:
        config_dir = Path(__file__).parents[1].resolve()

    def resolve_path(p: str | None, fallback: Path = None) -> Path:
        """Resolve a path relative to config file directory if not absolute."""
        if p is None:
            return fallback
        p = Path(p)
        return (config_dir / p).resolve() if not p.is_absolute() else p.resolve()

    args.data_dir = resolve_path(args.data_dir)
    args.csv_path = resolve_path(args.csv_path, fallback=args.data_dir / "dataset_split.csv")
    args.image_dir = resolve_path(args.image_dir, fallback=args.data_dir / "preprocessed_png_256")
    args.results_dir = f"results/{args.out_dir}"
    args.checkpoints_dir = f"checkpoints/{args.out_dir}"
    ensure_dir(args.results_dir)
    ensure_dir(args.checkpoints_dir)
    return args

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

config = load_config()

if __name__ == '__main__':
    print(config)
