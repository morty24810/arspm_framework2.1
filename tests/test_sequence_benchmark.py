from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.sequence_benchmark import (
    FAMILY_ORDER,
    FeatureStandardizer,
    SequenceBenchmarkConfig,
    build_grouped_folds,
    build_window_arrays,
    compute_health,
    load_sequence_pool,
    run_benchmark,
)


def make_small_cfg(tmp_path: Path) -> SequenceBenchmarkConfig:
    return SequenceBenchmarkConfig(
        families=("A2",),
        models=("GRU", "LSTM", "TCN", "ATTENTION"),
        kfolds=2,
        window=16,
        batch_size=16,
        epochs=1,
        patience=1,
        seed=42,
        outdir=str(tmp_path),
        device="cpu",
        hidden_dim=16,
        attention_layers=1,
        tcn_channels=(16, 16),
        max_sequences_per_family=4,
        max_windows_per_sequence=16,
    )


def test_sequence_ids_unique_and_a2_feed_asymmetry():
    cfg = SequenceBenchmarkConfig()
    sequence_map, summary = load_sequence_pool(cfg)
    assert len(sequence_map) == 100
    assert summary["sequence_ids_unique"] == 1
    assert summary["a2_feed_train_only"]
    assert summary["a2_feed_test_only"]
    families = {row["family"] for row in summary["rows"]}
    assert families == set(FAMILY_ORDER)


def test_health_formula_monotonic():
    dp = np.asarray([25.0, 100.0, 300.0, 600.0, 700.0], dtype=np.float32)
    health = compute_health(dp, clean_dp=25.0, failure_dp=600.0)
    assert np.all((0.0 <= health) & (health <= 1.0))
    assert np.all(np.diff(health) <= 1e-6)
    assert np.isclose(float(health[0]), 1.0)
    assert np.isclose(float(health[3]), 0.0)
    assert np.isclose(float(health[4]), 0.0)


def test_grouped_folds_do_not_leak_sequences(tmp_path: Path):
    cfg = make_small_cfg(tmp_path)
    sequence_map, _summary = load_sequence_pool(cfg)
    folds, selected_ids = build_grouped_folds("A2", sequence_map, cfg)
    selected_set = set(selected_ids)
    assert len(folds) == 2
    for fold in folds:
        train_ids = set(fold.train_ids)
        val_ids = set(fold.val_ids)
        test_ids = set(fold.test_ids)
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert train_ids | val_ids | test_ids == selected_set


def test_standardizer_uses_train_fold_statistics(tmp_path: Path):
    cfg = make_small_cfg(tmp_path)
    sequence_map, _summary = load_sequence_pool(cfg)
    folds, _selected_ids = build_grouped_folds("A2", sequence_map, cfg)
    train_x, _train_y, _train_meta = build_window_arrays(folds[0].train_ids, sequence_map, cfg)
    standardizer = FeatureStandardizer().fit(train_x)
    expected_mean = train_x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    expected_std = train_x.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    expected_std[expected_std < 1e-6] = 1.0
    assert np.allclose(standardizer.mean, expected_mean)
    assert np.allclose(standardizer.std, expected_std)


def test_smoke_run_generates_outputs_for_all_models(tmp_path: Path):
    cfg = make_small_cfg(tmp_path)
    result = run_benchmark(cfg)
    run_root = Path(result["out_root"])
    assert run_root.exists()
    assert (run_root / "metrics_by_fold.csv").exists()
    assert (run_root / "metrics_by_family.csv").exists()
    assert (run_root / "metrics_overall.csv").exists()
    assert (run_root / "metrics_overall.json").exists()
    assert (run_root / "model_rankings.csv").exists()
    assert (run_root / "plots" / "metric_bars_overall.png").exists()
    assert (run_root / "plots" / "scatter_health_pred.png").exists()
    assert (run_root / "plots" / "scatter_rul_pred.png").exists()

    fold_df = pd.read_csv(run_root / "metrics_by_fold.csv")
    assert set(fold_df["model"]) == {"GRU", "LSTM", "TCN", "ATTENTION"}
    for model_name in ("gru", "lstm", "tcn", "attention"):
        artifact_dir = run_root / "artifacts" / "A2" / model_name
        assert artifact_dir.exists()
        assert any(path.name.endswith("_best.pt") for path in artifact_dir.iterdir())
        prediction_csvs = sorted(artifact_dir.glob("fold_*_predictions.csv"))
        assert prediction_csvs
        prediction_df = pd.read_csv(prediction_csvs[0])
        train_rows = prediction_df[prediction_df["source_split"] == "train"]
        if not train_rows.empty:
            assert train_rows["rul_pred"].isna().all()
