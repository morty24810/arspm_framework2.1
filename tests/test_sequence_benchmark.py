from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.sequence_benchmark import (
    FAMILY_ORDER,
    FeatureStandardizer,
    SequenceBenchmarkConfig,
    apply_monotone_projection,
    build_grouped_folds,
    build_supervised_sample_weights,
    build_supervised_arrays,
    compute_dp_proxy_health,
    compute_health_remaining,
    family_train_sequence_ids,
    load_sequence_pool,
    pretrain_ids_for_fold,
    run_benchmark,
)


def make_small_cfg(tmp_path: Path) -> SequenceBenchmarkConfig:
    return SequenceBenchmarkConfig(
        families=("A2", "A3", "A4"),
        models=("LSTM",),
        kfolds=2,
        window=16,
        batch_size=16,
        epochs=1,
        patience=1,
        pretrain_epochs=1,
        pretrain_patience=1,
        seed=42,
        outdir=str(tmp_path),
        device="cpu",
        hidden_dim=16,
        attention_layers=1,
        tcn_channels=(16, 16),
        max_sequences_per_family=2,
        max_windows_per_sequence=8,
        tuning_windows=(16,),
        tuning_hidden_dims=(16,),
        tuning_dropouts=(0.10,),
    )


def make_compare_cfg(tmp_path: Path) -> SequenceBenchmarkConfig:
    return SequenceBenchmarkConfig(
        families=("A2",),
        models=("GRU", "LSTM", "TCN", "ATTENTION"),
        kfolds=2,
        window=16,
        batch_size=16,
        epochs=1,
        patience=1,
        pretrain_epochs=1,
        pretrain_patience=1,
        seed=42,
        outdir=str(tmp_path),
        device="cpu",
        hidden_dim=16,
        attention_layers=1,
        tcn_channels=(16, 16),
        max_sequences_per_family=4,
        max_windows_per_sequence=16,
        enable_family_tuning=False,
    )


def test_sequence_ids_unique_and_a2_feed_asymmetry():
    cfg = SequenceBenchmarkConfig()
    sequence_map, summary = load_sequence_pool(cfg)
    assert len(sequence_map) == 100
    assert cfg.models == ("LSTM",)
    assert summary["sequence_ids_unique"] == 1
    assert summary["health_target_type"] == "remaining_life_fraction"
    assert summary["a2_feed_train_only"]
    assert summary["a2_feed_test_only"]
    families = {row["family"] for row in summary["rows"]}
    assert families == set(FAMILY_ORDER)


def test_test_sequences_have_constant_total_life_and_monotone_remaining_health():
    cfg = SequenceBenchmarkConfig()
    sequence_map, _summary = load_sequence_pool(cfg)
    for sample in sequence_map.values():
        if sample.source_split != "test":
            continue
        assert np.allclose(sample.total_life, sample.total_life[0])
        assert np.all((0.0 <= sample.health_remaining_true) & (sample.health_remaining_true <= 1.0))
        assert np.all(np.diff(sample.health_remaining_true) <= 1e-6)
        assert not np.allclose(sample.dp_proxy_health, sample.health_remaining_true)


def test_health_helpers_have_expected_behavior():
    dp = np.asarray([25.0, 100.0, 300.0, 600.0, 700.0], dtype=np.float32)
    proxy = compute_dp_proxy_health(dp, clean_dp=25.0, failure_dp=600.0)
    assert np.all((0.0 <= proxy) & (proxy <= 1.0))
    assert np.all(np.diff(proxy) <= 1e-6)

    time_arr = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    rul = np.asarray([10.0, 9.9, 9.8, 9.7], dtype=np.float32)
    remaining = compute_health_remaining(time_arr, rul)
    assert np.all((0.0 <= remaining) & (remaining <= 1.0))
    assert np.all(np.diff(remaining) <= 1e-6)
    assert np.isclose(float(remaining[0]), 1.0)


def test_grouped_folds_use_test_sequences_only_and_pretraining_excludes_val_test(tmp_path: Path):
    cfg = make_small_cfg(tmp_path)
    sequence_map, _summary = load_sequence_pool(cfg)
    folds, selected_test_ids = build_grouped_folds("A2", sequence_map, cfg)
    train_ids = set(family_train_sequence_ids("A2", sequence_map, cfg))
    assert len(folds) == 2
    assert all(sequence_map[sid].source_split == "test" for sid in selected_test_ids)
    for fold in folds:
        train_fold_ids = set(fold.train_ids)
        val_ids = set(fold.val_ids)
        test_ids = set(fold.test_ids)
        assert train_fold_ids.isdisjoint(val_ids)
        assert train_fold_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert all(sequence_map[sid].source_split == "test" for sid in train_fold_ids | val_ids | test_ids)
        pretrain_ids = set(pretrain_ids_for_fold("A2", fold, sequence_map, cfg))
        assert train_ids.issubset(pretrain_ids)
        assert val_ids.isdisjoint(pretrain_ids)
        assert test_ids.isdisjoint(pretrain_ids)


def test_standardizer_uses_pretrain_statistics(tmp_path: Path):
    cfg = make_compare_cfg(tmp_path)
    sequence_map, _summary = load_sequence_pool(cfg)
    folds, _selected_test_ids = build_grouped_folds("A2", sequence_map, cfg)
    train_x, _train_y, _train_meta = build_supervised_arrays(folds[0].train_ids, sequence_map, cfg)
    standardizer = FeatureStandardizer().fit(train_x)
    expected_mean = train_x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    expected_std = train_x.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    expected_std[expected_std < 1e-6] = 1.0
    assert np.allclose(standardizer.mean, expected_mean)
    assert np.allclose(standardizer.std, expected_std)


def test_weighting_and_a2_long_life_boost(tmp_path: Path):
    cfg = make_compare_cfg(tmp_path)
    sequence_map, _summary = load_sequence_pool(cfg)
    a3_folds, _ = build_grouped_folds("A3", sequence_map, cfg)
    _train_x, _train_y, a3_meta = build_supervised_arrays(a3_folds[0].train_ids, sequence_map, cfg)
    a3_weights = build_supervised_sample_weights("A3", a3_meta, a3_folds[0].train_ids, sequence_map, cfg)
    a3_meta = a3_meta.copy()
    a3_meta["sample_weight"] = a3_weights
    low_cut = float(a3_meta["health_remaining_true"].quantile(0.25))
    high_cut = float(a3_meta["health_remaining_true"].quantile(0.75))
    late = a3_meta[a3_meta["health_remaining_true"] <= low_cut]["sample_weight"].mean()
    early = a3_meta[a3_meta["health_remaining_true"] >= high_cut]["sample_weight"].mean()
    assert late > early

    a2_folds, _ = build_grouped_folds("A2", sequence_map, cfg)
    _train_x, _train_y, a2_meta = build_supervised_arrays(a2_folds[0].train_ids, sequence_map, cfg)
    a2_weights = build_supervised_sample_weights("A2", a2_meta, a2_folds[0].train_ids, sequence_map, cfg)
    a2_meta = a2_meta.copy()
    a2_meta["sample_weight"] = a2_weights
    seq_life = sorted(
        [(sequence_id, float(sequence_map[sequence_id].total_life_scalar)) for sequence_id in a2_folds[0].train_ids],
        key=lambda item: item[1],
    )
    long_ids = {sequence_id for sequence_id, _ in seq_life[-max(1, int(np.ceil(len(seq_life) / 3.0))):]}
    if long_ids:
        long_mean = a2_meta[a2_meta["sequence_id"].isin(long_ids)]["sample_weight"].mean()
        short_mean = a2_meta[~a2_meta["sequence_id"].isin(long_ids)]["sample_weight"].mean()
        if np.isfinite(short_mean):
            assert long_mean > short_mean
        else:
            assert np.isfinite(long_mean) and long_mean > 0.0


def test_monotone_projection_adds_raw_columns_and_projects_nonincreasing():
    pred_df = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1", "s2", "s2"],
            "time": [0.1, 0.2, 0.3, 0.1, 0.2],
            "total_life": [10.0, 10.0, 10.0, 5.0, 5.0],
            "health_remaining_pred": [0.9, 0.8, 0.85, 0.7, 0.6],
            "rul_pred": [9.0, 8.0, 8.5, 3.5, 3.0],
            "is_observed": [True, True, True, True, True],
            "is_tail_reference": [False, False, False, False, False],
        }
    )
    projected = apply_monotone_projection(pred_df, cfg=SequenceBenchmarkConfig(enable_family_tuning=False))
    assert {"health_remaining_pred_raw", "rul_pred_raw"}.issubset(projected.columns)
    seq = projected[projected["sequence_id"] == "s1"].sort_values("time")
    assert np.all(np.diff(seq["health_remaining_pred"].to_numpy(dtype=np.float32)) <= 1e-6)
    assert np.allclose(
        seq["rul_pred"].to_numpy(dtype=np.float32),
        seq["health_remaining_pred"].to_numpy(dtype=np.float32) * seq["total_life"].to_numpy(dtype=np.float32),
    )
    assert np.allclose(seq["health_remaining_pred_raw"].to_numpy(dtype=np.float32), np.asarray([0.9, 0.8, 0.85], dtype=np.float32))


def test_smoke_run_generates_lstm_default_outputs(tmp_path: Path):
    cfg = make_small_cfg(tmp_path)
    result = run_benchmark(cfg)
    run_root = Path(result["out_root"])
    assert run_root.exists()
    assert (run_root / "metrics_by_fold.csv").exists()
    assert (run_root / "metrics_by_family.csv").exists()
    assert (run_root / "metrics_overall.csv").exists()
    assert (run_root / "metrics_overall.json").exists()
    assert (run_root / "model_rankings.csv").exists()
    assert (run_root / "phase1_protocol" / "metrics_by_family.csv").exists()
    assert (run_root / "family_tuning_search.csv").exists()
    assert (run_root / "selected_lstm_family_configs.csv").exists()
    assert (run_root / "protocol_vs_tuned_compare.csv").exists()
    assert (run_root / "baseline_compare_phase1.json").exists()
    assert (run_root / "baseline_compare_phase2.json").exists()
    assert (run_root / "acceptance_summary.json").exists()
    assert (run_root / "plots" / "metric_bars_overall.png").exists()
    assert (run_root / "plots" / "scatter_health_remaining_pred.png").exists()
    assert (run_root / "plots" / "scatter_rul_pred.png").exists()

    fold_df = pd.read_csv(run_root / "metrics_by_fold.csv")
    assert set(fold_df["model"]) == {"LSTM"}
    prediction_csvs = sorted((run_root / "artifacts" / "A3" / "lstm").glob("fold_*_predictions.csv"))
    assert prediction_csvs
    prediction_df = pd.read_csv(prediction_csvs[0])
    required_cols = {
        "health_remaining_true",
        "health_remaining_pred",
        "health_remaining_pred_raw",
        "dp_proxy_health",
        "rul_true",
        "rul_pred",
        "rul_pred_raw",
        "total_life",
        "is_observed",
        "is_tail_reference",
    }
    assert required_cols.issubset(prediction_df.columns)
    observed = prediction_df[prediction_df["is_observed"] == True].copy()
    assert np.allclose(
        observed["rul_pred"].to_numpy(dtype=np.float32),
        observed["health_remaining_pred"].to_numpy(dtype=np.float32) * observed["total_life"].to_numpy(dtype=np.float32),
    )
    for _sid, group in observed.groupby("sequence_id"):
        seq = group.sort_values("time")
        assert np.all(np.diff(seq["health_remaining_pred"].to_numpy(dtype=np.float32)) <= 1e-6)


def test_smoke_run_compare_mode_keeps_all_models(tmp_path: Path):
    cfg = make_compare_cfg(tmp_path)
    result = run_benchmark(cfg)
    run_root = Path(result["out_root"])
    fold_df = pd.read_csv(run_root / "metrics_by_fold.csv")
    assert set(fold_df["model"]) == {"GRU", "LSTM", "TCN", "ATTENTION"}
    for model_name in ("gru", "lstm", "tcn", "attention"):
        artifact_dir = run_root / "artifacts" / "A2" / model_name
        assert artifact_dir.exists()
