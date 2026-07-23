"""執行可重現的 ChiFraud 簡繁雙模型實驗。"""
from __future__ import annotations

import argparse
import json

from src.chifraud_experiment import run_dual_script_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="訓練並交叉驗收 ChiFraud 簡體／繁體 BERT")
    parser.add_argument("prepared_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--base-model", default="google-bert/bert-base-chinese")
    parser.add_argument("--base-revision", required=True, help="Hugging Face commit SHA；本機模型可填 local")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    report = run_dual_script_experiment(
        args.prepared_dir,
        args.output_dir,
        base_model=args.base_model,
        base_revision=args.base_revision,
        seeds=args.seeds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
