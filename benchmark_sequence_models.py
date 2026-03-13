from __future__ import annotations

import argparse

from src.sequence_benchmark import SequenceBenchmarkConfig, run_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark family-specific sequence models on remaining-life health targets.")
    parser.add_argument("--train-csv", default="Train_Data_CSV.csv")
    parser.add_argument("--test-csv", default="Test_Data_CSV.csv")
    parser.add_argument("--models", default="GRU,LSTM,TCN,ATTENTION")
    parser.add_argument("--families", default="A2,A3,A4")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="outputs/sequence_model_benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-sequences-per-family", type=int, default=None)
    parser.add_argument("--max-windows-per-sequence", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SequenceBenchmarkConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        models=tuple(item.strip() for item in args.models.split(",") if item.strip()),
        families=tuple(item.strip() for item in args.families.split(",") if item.strip()),
        kfolds=int(args.kfolds),
        window=int(args.window),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        pretrain_epochs=int(args.pretrain_epochs),
        pretrain_patience=int(args.pretrain_patience),
        seed=int(args.seed),
        outdir=str(args.outdir),
        device=str(args.device),
        learning_rate=float(args.learning_rate),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        max_sequences_per_family=args.max_sequences_per_family,
        max_windows_per_sequence=args.max_windows_per_sequence,
    )
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
