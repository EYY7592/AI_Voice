import json


def test_dual_script_experiment_trains_and_cross_evaluates_both_candidates(tmp_path, monkeypatch) -> None:
    from src import chifraud_experiment

    calls = {"train": [], "evaluate": []}

    def fake_train(prepared_dir, output_dir, **kwargs):
        calls["train"].append((str(output_dir), kwargs["script_view"], kwargs["seed"]))
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True)
        return {"status": "passed", "script_view": kwargs["script_view"], "seed": kwargs["seed"]}

    def fake_evaluate(model_dir, prepared_dir, **kwargs):
        candidate = model_dir.parent.name.split("-seed-")[0]
        calls["evaluate"].append((candidate, kwargs["script_view"]))
        offset = 0.004 if candidate == "simplified" else 0.0
        return {
            "passed": True,
            "years": {
                "2022": {"fraud_recall": 0.91 + offset, "macro_f1": 0.90 + offset},
                "2023": {"fraud_recall": 0.92 + offset, "macro_f1": 0.91 + offset},
            },
        }

    monkeypatch.setattr(chifraud_experiment, "train_candidate", fake_train)
    monkeypatch.setattr(chifraud_experiment, "evaluate_candidate_artifact", fake_evaluate)

    report = chifraud_experiment.run_dual_script_experiment(
        tmp_path / "prepared",
        tmp_path / "experiment",
        base_model="google-bert/bert-base-chinese",
        base_revision="pinned-sha",
        seeds=(42,),
    )

    assert len(calls["train"]) == 2
    assert len(calls["evaluate"]) == 4
    assert report["selection"]["candidate"] == "traditional"
    assert report["selection"]["reason"] == "equivalent"
    saved = json.loads((tmp_path / "experiment" / "selection_report.json").read_text(encoding="utf-8"))
    assert saved["base_revision"] == "pinned-sha"


def test_dual_script_experiment_stops_selection_when_candidate_fails(tmp_path, monkeypatch) -> None:
    from src import chifraud_experiment

    def fake_train(prepared_dir, output_dir, **kwargs):
        return {"status": "failed", "failures": ["Recall 未達標"], "script_view": kwargs["script_view"]}

    monkeypatch.setattr(chifraud_experiment, "train_candidate", fake_train)

    report = chifraud_experiment.run_dual_script_experiment(
        tmp_path / "prepared",
        tmp_path / "experiment",
        base_model="local-model",
        seeds=(42,),
    )

    assert report["selection"]["status"] == "failed"
    assert report["cross_evaluations"] == []
