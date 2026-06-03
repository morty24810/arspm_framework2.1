from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import SimConfig, resolve_machine_set
from src.rul_predictor import _PaperGRURegressor


def build_windows(values: np.ndarray, window: int) -> np.ndarray:
    seq = np.asarray(values, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq[:, None]
    n, f = seq.shape
    windows = np.zeros((n, window, f), dtype=np.float32)
    for idx in range(n):
        start = idx - window + 1
        if start < 0:
            pad = np.repeat(seq[0:1], repeats=-start, axis=0)
            core = seq[0 : idx + 1]
            windows[idx] = np.concatenate([pad, core], axis=0)
        else:
            windows[idx] = seq[start : idx + 1]
    return windows


def train_paper_gru_fit(cfg: SimConfig, data_no: int) -> tuple[dict, pd.DataFrame, dict, dict]:
    test_csv = Path(cfg.TEST_CSV)
    if not test_csv.exists():
        raise FileNotFoundError(f"missing TEST_CSV: {test_csv}")

    features = [str(f) for f in getattr(cfg, "RUL_FEATURES", ("Differential_pressure",))]
    df = pd.read_csv(test_csv)
    required = ["Data_No", "Time", "RUL", *features]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {test_csv}: {missing}")

    g = df[df["Data_No"] == int(data_no)].copy()
    if g.empty:
        raise ValueError(f"Data_No={data_no} not found in {test_csv}")
    g = g.sort_values("Time").reset_index(drop=True)

    window = int(getattr(cfg, "RUL_WINDOW", 30))
    val_ratio = float(getattr(cfg, "RUL_VAL_RATIO", 0.2))
    batch_size = int(getattr(cfg, "RUL_GRU_BATCH_SIZE", 1024))
    epochs = int(getattr(cfg, "RUL_GRU_EPOCHS", 250))
    lr = float(getattr(cfg, "RUL_GRU_LR", 1e-3))
    hidden_dim = int(getattr(cfg, "RUL_GRU_HIDDEN_DIM", 40))
    num_layers = int(getattr(cfg, "RUL_GRU_NUM_LAYERS", 1))
    dropout = float(getattr(cfg, "RUL_GRU_DROPOUT", 0.25))
    seed = int(getattr(cfg, "SEED", 42))

    split_idx = int(round(len(g) * (1.0 - val_ratio)))
    split_idx = min(max(split_idx, window), len(g) - 1)

    raw_x = g[features].to_numpy(dtype=np.float32)
    total_life = float((g["Time"] + g["RUL"]).iloc[0])
    y_abs = g["RUL"].to_numpy(dtype=np.float32)
    y_norm = np.clip(y_abs / max(total_life, 1e-6), 0.0, 1.0)

    mean = raw_x[:split_idx].mean(axis=0, keepdims=True).astype(np.float32)
    std = raw_x[:split_idx].std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    norm_x = (raw_x - mean) / std
    windows = build_windows(norm_x, window)

    x_train = torch.tensor(windows[:split_idx], dtype=torch.float32)
    y_train = torch.tensor(y_norm[:split_idx], dtype=torch.float32)
    x_val = torch.tensor(windows[split_idx:], dtype=torch.float32)
    y_val = torch.tensor(y_norm[split_idx:], dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(batch_size, max(len(x_train), 1)),
        shuffle=True,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _PaperGRURegressor(len(features), hidden_dim, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    patience = max(5, epochs // 4)
    stale = 0
    history = {"epoch": [], "train_mse": [], "val_mse": []}

    for epoch in range(1, max(epochs, 1) + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            train_pred = model(x_train)
            val_pred = model(x_val)
            train_loss = float(criterion(train_pred, y_train).item())
            val_loss = float(criterion(val_pred, y_val).item())

        history["epoch"].append(epoch)
        history["train_mse"].append(train_loss if batch_losses else 0.0)
        history["val_mse"].append(val_loss)

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        full_pred_norm = np.asarray(
            model(torch.tensor(windows, dtype=torch.float32)).detach().cpu().tolist(),
            dtype=np.float32,
        )
    pred_abs = np.clip(full_pred_norm, 0.0, 1.0) * total_life
    residual_abs = pred_abs - y_abs

    split_labels = np.where(np.arange(len(g)) < split_idx, "train", "val")
    pred_df = pd.DataFrame(
        {
            "Data_No": int(data_no),
            "Time": g["Time"].to_numpy(dtype=np.float32),
            "true_rul": y_abs,
            "pred_rul": pred_abs.astype(np.float32),
            "residual_rul": residual_abs.astype(np.float32),
            "split": split_labels,
        }
    )

    train_mask = pred_df["split"] == "train"
    val_mask = pred_df["split"] == "val"

    def _rmse(mask):
        arr = pred_df.loc[mask, "residual_rul"].to_numpy(dtype=np.float64)
        return float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0

    def _mae(mask):
        arr = pred_df.loc[mask, "residual_rul"].to_numpy(dtype=np.float64)
        return float(np.mean(np.abs(arr))) if arr.size else 0.0

    summary = {
        "test_csv": str(test_csv),
        "data_no": int(data_no),
        "features": features,
        "window": int(window),
        "split_idx": int(split_idx),
        "sequence_length": int(len(g)),
        "train_points": int(train_mask.sum()),
        "val_points": int(val_mask.sum()),
        "total_life": float(total_life),
        "rul_start": float(y_abs[0]),
        "rul_end": float(y_abs[-1]),
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "train_rmse_rul": _rmse(train_mask),
        "val_rmse_rul": _rmse(val_mask),
        "train_mae_rul": _mae(train_mask),
        "val_mae_rul": _mae(val_mask),
        "seed": int(seed),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "epochs_configured": int(epochs),
        "epochs_ran": int(len(history["epoch"])),
        "batch_size": int(batch_size),
        "lr": float(lr),
    }
    fit_state = {
        "model": model,
        "mean": mean.squeeze(0),
        "std": std.squeeze(0),
        "features": features,
        "window": int(window),
        "train_data_no": int(data_no),
        "test_csv": str(test_csv),
    }
    return summary, pred_df, history, fit_state


def predict_machine_trajectory(
    model: _PaperGRURegressor,
    mean: np.ndarray,
    std: np.ndarray,
    window: int,
    features: list[str],
    machine_df: pd.DataFrame,
) -> pd.DataFrame:
    g = machine_df.sort_values("Time").reset_index(drop=True).copy()
    raw_x = g[features].to_numpy(dtype=np.float32)
    total_life = float((g["Time"] + g["RUL"]).iloc[0])
    true_norm = np.clip(g["RUL"].to_numpy(dtype=np.float32) / max(total_life, 1e-6), 0.0, 1.0)
    norm_x = (raw_x - mean[None, :]) / std[None, :]
    windows = build_windows(norm_x, window)
    with torch.no_grad():
        pred_norm = np.asarray(
            model(torch.tensor(windows, dtype=torch.float32)).detach().cpu().tolist(),
            dtype=np.float32,
        )
    pred_norm = np.clip(pred_norm, 0.0, 1.0)
    pred_abs = pred_norm * total_life
    pred_norm_aligned = pred_norm / max(float(pred_norm[0]), 1e-6)
    pred_norm_aligned = np.clip(pred_norm_aligned, 0.0, 1.0)
    true_norm_aligned = true_norm / max(float(true_norm[0]), 1e-6)
    true_norm_aligned = np.clip(true_norm_aligned, 0.0, 1.0)
    return pd.DataFrame(
        {
            "Data_No": int(g["Data_No"].iloc[0]),
            "Time": g["Time"].to_numpy(dtype=np.float32),
            "true_rul": g["RUL"].to_numpy(dtype=np.float32),
            "pred_rul": pred_abs.astype(np.float32),
            "true_rul_norm": true_norm.astype(np.float32),
            "pred_rul_norm": pred_norm.astype(np.float32),
            "true_rul_norm_aligned": true_norm_aligned.astype(np.float32),
            "pred_rul_norm_aligned": pred_norm_aligned.astype(np.float32),
            "total_life": float(total_life),
        }
    )


def save_machine_set_panel(
    outdir: Path,
    machine_preds: list[pd.DataFrame],
    *,
    machine_set_label: str,
):
    if not machine_preds:
        return

    n = len(machine_preds)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes = axes.reshape(rows, cols)

    summary_rows = []
    for idx, pred_df in enumerate(machine_preds):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        machine_idx = idx + 1
        data_no = int(pred_df["Data_No"].iloc[0])
        ax.plot(pred_df["Time"], pred_df["pred_rul_norm_aligned"], color="#4c9ad4", linewidth=1.7, label="Pred RUL by GRU")
        ax.plot(pred_df["Time"], pred_df["true_rul_norm_aligned"], color="#f4a261", linewidth=1.4, label="Actual RUL")
        ax.set_title(f"Machine {machine_idx} (Data_No={data_no})", fontsize=11)
        ax.set_xlabel("Machine running time")
        if c == 0:
            ax.set_ylabel("RUL")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

        resid = pred_df["pred_rul"] - pred_df["true_rul"]
        rmse = float(np.sqrt(np.mean(np.square(resid))))
        mae = float(np.mean(np.abs(resid)))
        summary_rows.append(
            {
                "machine_index": machine_idx,
                "Data_No": data_no,
                "length": int(len(pred_df)),
                "time_min": float(pred_df["Time"].min()),
                "time_max": float(pred_df["Time"].max()),
                "total_life": float(pred_df["total_life"].iloc[0]),
                "pred_start_raw": float(pred_df["pred_rul_norm"].iloc[0]),
                "rmse_rul": rmse,
                "mae_rul": mae,
            }
        )

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Predicted normalized RUL trajectories of experiment machines ({machine_set_label})",
        fontsize=14,
        y=1.02,
    )
    fig_path = outdir / f"rul_gru_predicted_machines_{machine_set_label}.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.concat(machine_preds, ignore_index=True).to_csv(
        outdir / f"rul_gru_predicted_machines_{machine_set_label}.csv",
        index=False,
    )
    pd.DataFrame(summary_rows).to_csv(
        outdir / f"rul_gru_predicted_machines_{machine_set_label}_summary.csv",
        index=False,
    )


def save_figures(outdir: Path, summary: dict, pred_df: pd.DataFrame, history: dict, feature_series: pd.Series):
    outdir.mkdir(parents=True, exist_ok=True)
    split_time = float(pred_df["Time"].iloc[int(summary["split_idx"])]) if int(summary["split_idx"]) < len(pred_df) else float(pred_df["Time"].iloc[-1])

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 11),
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.2]},
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(pred_df["Time"], pred_df["true_rul"], color="#1f4e79", linewidth=2.2, label="True RUL")
    ax.plot(pred_df["Time"], pred_df["pred_rul"], color="#d94841", linewidth=1.8, label="Predicted RUL")
    ax.axvspan(float(pred_df["Time"].iloc[0]), split_time, color="#d8ecff", alpha=0.25, label="Train segment")
    ax.axvspan(split_time, float(pred_df["Time"].iloc[-1]), color="#ffe7d6", alpha=0.22, label="Validation segment")
    ax.axvline(split_time, color="#7a7a7a", linestyle="--", linewidth=1.2)
    ax.set_title(
        f"Paper-GRU fit disclosure on Test_Data_CSV Data_No={summary['data_no']}\n"
        f"feature={','.join(summary['features'])}, window={summary['window']}, best epoch={summary['best_epoch']}"
    )
    ax.set_ylabel("RUL")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2, frameon=False)

    ax = axes[1]
    ax.plot(pred_df["Time"], feature_series.to_numpy(dtype=np.float32), color="#2a9d8f", linewidth=1.4)
    ax.axvline(split_time, color="#7a7a7a", linestyle="--", linewidth=1.2)
    ax.set_ylabel(summary["features"][0])
    ax.set_title("Observed health-related input feature used by the GRU")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(pred_df["Time"], pred_df["residual_rul"], color="#6a4c93", linewidth=1.4)
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.axvline(split_time, color="#7a7a7a", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Prediction error")
    ax.set_title(
        f"Residuals: train RMSE={summary['train_rmse_rul']:.3f}, val RMSE={summary['val_rmse_rul']:.3f}"
    )
    ax.grid(alpha=0.25)

    overview_path = outdir / f"rul_gru_fit_summary_data_no_{summary['data_no']}.png"
    fig.savefig(overview_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.plot(history["epoch"], history["train_mse"], color="#1f4e79", linewidth=1.8, label="Train MSE")
    ax.plot(history["epoch"], history["val_mse"], color="#d94841", linewidth=1.8, label="Validation MSE")
    ax.axvline(int(summary["best_epoch"]), color="#7a7a7a", linestyle="--", linewidth=1.0, label="Best epoch")
    ax.set_title("Paper-GRU optimization history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    loss_path = outdir / f"rul_gru_fit_losses_data_no_{summary['data_no']}.png"
    fig.savefig(loss_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    axes[0].scatter(pred_df["true_rul"], pred_df["pred_rul"], s=10, alpha=0.6, color="#2a9d8f")
    lim = [0.0, max(float(pred_df["true_rul"].max()), float(pred_df["pred_rul"].max()))]
    axes[0].plot(lim, lim, linestyle="--", color="#666666", linewidth=1.0)
    axes[0].set_title("True vs predicted RUL")
    axes[0].set_xlabel("True RUL")
    axes[0].set_ylabel("Predicted RUL")
    axes[0].grid(alpha=0.25)

    axes[1].hist(pred_df["residual_rul"], bins=40, color="#e76f51", alpha=0.85, edgecolor="white")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Prediction error")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.2)
    residual_path = outdir / f"rul_gru_fit_residuals_data_no_{summary['data_no']}.png"
    fig.savefig(residual_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Disclose paper-style GRU RUL fitting results.")
    parser.add_argument("--data-no", type=int, default=None, help="Machine/Data_No to fit and visualize.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/rul_gru_fit",
        help="Directory used to save figures, CSV, and JSON summaries.",
    )
    parser.add_argument(
        "--panel-machine-set",
        type=str,
        default="current6",
        choices=["current6", "paper8"],
        help="Machine set used for the multi-machine paper-style RUL panel.",
    )
    args = parser.parse_args()

    cfg = SimConfig()
    data_no = int(args.data_no if args.data_no is not None else getattr(cfg, "RUL_TRAIN_DATA_NO", 18))
    outdir = Path(args.outdir) / f"data_no_{data_no}"
    outdir.mkdir(parents=True, exist_ok=True)

    summary, pred_df, history, fit_state = train_paper_gru_fit(cfg, data_no)
    source = pd.read_csv(cfg.TEST_CSV)
    feature_name = summary["features"][0]
    feature_series = source[source["Data_No"] == data_no].sort_values("Time")[feature_name].reset_index(drop=True)

    pred_csv = outdir / f"rul_gru_fit_predictions_data_no_{data_no}.csv"
    summary_json = outdir / f"rul_gru_fit_summary_data_no_{data_no}.json"
    history_csv = outdir / f"rul_gru_fit_history_data_no_{data_no}.csv"

    pred_df.to_csv(pred_csv, index=False)
    pd.DataFrame(history).to_csv(history_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    save_figures(outdir, summary, pred_df, history, feature_series)

    panel_cfg = SimConfig()
    panel_cfg.MACHINE_SET_MODE = str(args.panel_machine_set)
    machine_set_label, machine_ids, _ = resolve_machine_set(panel_cfg)
    machine_preds = []
    source_sorted = pd.read_csv(cfg.TEST_CSV)
    for machine_id in machine_ids:
        g = source_sorted[source_sorted["Data_No"] == int(machine_id)].copy()
        if g.empty:
            continue
        machine_preds.append(
            predict_machine_trajectory(
                fit_state["model"],
                np.asarray(fit_state["mean"], dtype=np.float32),
                np.asarray(fit_state["std"], dtype=np.float32),
                int(fit_state["window"]),
                list(fit_state["features"]),
                g,
            )
        )
    save_machine_set_panel(outdir, machine_preds, machine_set_label=machine_set_label)

    print(f"saved RUL GRU fit disclosure to {outdir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
