"""簡體／繁體 ChiFraud 候選模型的成對實驗編排。"""
from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Sequence

from src.chifraud_training import (
    evaluate_candidate_artifact,
    select_script_candidate,
    train_candidate,
)


def _write_report(output: Path, report: dict[str, object]) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_dual_script_experiment(
    prepared_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    base_model: str | PathLike[str],
    base_revision: str | None = None,
    seeds: Sequence[int] = (42,),
    **training_options: object,
) -> dict[str, object]:
    """訓練兩種字體候選，並在兩種 test view 上做成對比較。"""
    normalized_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not normalized_seeds or len(normalized_seeds) > 3:
        raise ValueError("seeds 必須包含 1 到 3 個不重複整數。")
    output = Path(output_dir)
    prepared = Path(prepared_dir)
    manifests: list[dict[str, object]] = []
    candidate_paths: dict[tuple[str, int], Path] = {}

    for seed in normalized_seeds:
        for script_view in ("simplified", "traditional"):
            candidate_output = output / f"{script_view}-seed-{seed}"
            manifest = train_candidate(
                prepared,
                candidate_output,
                base_model=base_model,
                base_revision=base_revision,
                script_view=script_view,
                seed=seed,
                **training_options,
            )
            manifests.append(manifest)
            candidate_paths[(script_view, seed)] = candidate_output / "model"

    failed = [manifest for manifest in manifests if manifest.get("status") != "passed"]
    if failed:
        return _write_report(
            output,
            {
                "base_model": str(base_model),
                "base_revision": base_revision,
                "seeds": list(normalized_seeds),
                "candidate_manifests": manifests,
                "cross_evaluations": [],
                "selection": {
                    "status": "failed",
                    "candidate": None,
                    "reason": "candidate_acceptance_failed",
                },
            },
        )

    cross_evaluations: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for seed in normalized_seeds:
        for candidate in ("simplified", "traditional"):
            for script_view in ("simplified", "traditional"):
                evaluation = evaluate_candidate_artifact(
                    candidate_paths[(candidate, seed)],
                    prepared,
                    script_view=script_view,
                )
                cross_evaluations.append(
                    {
                        "candidate": candidate,
                        "view": script_view,
                        "seed": seed,
                        "report": evaluation,
                    }
                )
                for year in (2022, 2023):
                    metrics = evaluation["years"][str(year)]
                    selection_rows.append(
                        {
                            "candidate": candidate,
                            "view": script_view,
                            "year": year,
                            "seed": seed,
                            "fraud_recall": metrics["fraud_recall"],
                            "macro_f1": metrics["macro_f1"],
                        }
                    )

    return _write_report(
        output,
        {
            "base_model": str(base_model),
            "base_revision": base_revision,
            "seeds": list(normalized_seeds),
            "candidate_manifests": manifests,
            "cross_evaluations": cross_evaluations,
            "selection_rows": selection_rows,
            "selection": select_script_candidate(selection_rows),
        },
    )
