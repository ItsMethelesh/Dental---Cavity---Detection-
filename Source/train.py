from __future__ import annotations

import argparse
import json

try:
    from cavity_detector.train import train_model
except ModuleNotFoundError:
    train_model = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for dental cavity detection.")
    parser.add_argument("--data-dir", type=str, default="data", help="Folder containing X-ray images.")
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="labels.csv",
        help="Optional CSV with image,label columns. If missing, pseudo labels are used.",
    )
    parser.add_argument("--output", type=str, default="models/cavity_cnn.pt", help="Model output path.")
    parser.add_argument("--image-size", type=int, default=224, help="Square input image size.")
    parser.add_argument("--epochs", type=int, default=14, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument(
        "--pseudo-quantile",
        type=float,
        default=0.62,
        help="Quantile threshold for pseudo labels when labels CSV is absent.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to train on (0 means all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if train_model is None:
        raise SystemExit(
            "PyTorch is not installed. Run: python -m pip install -r requirements.txt"
        )

    summary = train_model(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        model_output_path=args.output,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        pseudo_label_quantile=args.pseudo_quantile,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
