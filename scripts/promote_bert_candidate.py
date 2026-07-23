"""驗證並切換 localhost 使用的 BERT 候選模型。"""
from __future__ import annotations

import argparse
import json

from src.chifraud_promotion import promote_candidate, validate_candidate_output


def main() -> None:
    parser = argparse.ArgumentParser(description="驗證或升級 ChiFraud BERT candidate")
    parser.add_argument("candidate_dir", help="包含 model/ 與 training_manifest.json 的候選目錄")
    parser.add_argument("--target", default="models/bert_fraud")
    parser.add_argument("--promote", action="store_true", help="通過驗證後執行可回復的模型切換")
    parser.add_argument("--selection-report", help="實驗輸出的 selection_report.json")
    args = parser.parse_args()
    if args.promote and not args.selection_report:
        parser.error("--promote 必須同時提供 --selection-report。")
    result = (
        promote_candidate(
            args.candidate_dir,
            args.target,
            selection_report_path=args.selection_report,
        )
        if args.promote
        else validate_candidate_output(args.candidate_dir)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
