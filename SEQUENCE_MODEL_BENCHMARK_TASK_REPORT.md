# 時序模型 Benchmark 任務報告

## 1. 任務目標

這一批工作延續 v2 的 `remaining-life health` 目標，但把 benchmark 從「廣泛模型比較工具」收斂成「以 `LSTM` 為主線的優化流程」。

這一批的實際目標是：

- 優先把 `A3` 的表現拉近 `A4`
- 同時避免 `A2` 與 `A4` 發生退化
- 保留 `GRU / TCN / ATTENTION` 供後續比較，但不讓它們成為預設主路徑

## 2. 本批次已完成內容

### 2.1 預設 benchmark 路徑改為只跑 LSTM

- `benchmark_sequence_models.py` 的預設值現在是：
  - `--models LSTM`
- `GRU`、`TCN`、`ATTENTION` 都仍然保留
- 它們沒有被刪除
- 只有在明確指定 `--models` 時才會一起執行

### 2.2 fine-tuning 改用加權監督，而不是均勻 window loss

現在的 fine-tuning loss 不再是對所有 window 做無權平均的 MSE。

每個 supervised window 現在都會取得以下權重：

- sequence balancing：
  - `w_seq = 1 / windows_in_sequence`
- stage-aware weighting：
  - `w_stage = 1 + alpha_family * (1 - health_remaining_true)`

目前的 family-specific stage weights 為：

- `A2 = 1.0`
- `A3 = 1.5`
- `A4 = 1.0`

這代表：

- 長序列不會只因為產生更多 windows 就主導訓練
- 壽命中後段的 windows 會被賦予更高權重
- `A3` 在 late-life 區間獲得最強的強化

### 2.3 A2 新增明確的稀有子群補償路徑

外層 grouped `5-fold` 評估規則完全不變。

只在 `A2` 的訓練過程中：

- 依 `total_life` 對訓練序列排序
- 對最長壽命 tertile 加上額外倍率
- 目前倍率為：
  - `2.0`

這個補丁的目的，是在不改動正式評估切分的前提下，補償 `A2` 的低 feed 長壽命稀有子群。

### 2.4 單調投影已納入正式 benchmark 輸出

現在的 prediction 會同時保留兩種版本：

- raw：
  - `health_remaining_pred_raw`
  - `rul_pred_raw`
- projected：
  - `health_remaining_pred`
  - `rul_pred`

projected 版本會對每條 observed sequence 套用隨時間非增的 cumulative-minimum 約束。

目前的官方 metrics、plots 與 summaries 都以 projected prediction 為準。

這樣做的目的，是在保留 raw output 供審計的同時，讓 benchmark 主輸出更符合 remaining-life semantics。

### 2.5 已實作 family-specific 的 LSTM tuning

當 active model list 只有 `LSTM` 時，benchmark 現在會自動跑兩個階段：

1. protocol baseline pass
   - 三個 family 共用同一組 `LSTM` 配置
2. family tuning + final pass
   - 搜尋網格：
     - `window ∈ {32, 48}`
     - `hidden_dim ∈ {32, 64}`
     - `dropout ∈ {0.05, 0.10}`
   - 每個 family 各自選出最佳配置
   - 再用選中的 family-specific 配置重跑最終 benchmark

若使用者一次指定多個模型，family tuning 會自動跳過，benchmark 會回到一般 compare 模式。

### 2.6 已實作 baseline compare 與 acceptance reporting

目前鎖定的參考基線為：

- `outputs/sequence_model_benchmark/20260313_141821`

每次優化版 `LSTM` run 都會額外輸出：

- `baseline_compare_phase1.csv/json`
- `baseline_compare_phase2.csv/json`
- `protocol_vs_tuned_compare.csv`
- `acceptance_summary.json`
- `selected_lstm_family_configs.csv/json`
- `family_tuning_search.csv`

這些檔案用來明確回答：

- `A3` 有沒有提升
- `A2` 有沒有退化
- `A4` 有沒有退化
- worst-fold `RUL` 是變好還是變差

## 3. 目前程式的實際行為

獨立入口仍然是：

- `benchmark_sequence_models.py`

可重用模組仍然是：

- `src/sequence_benchmark.py`

主要監督目標仍然是：

- `health_remaining = RUL / total_life`

benchmark 內部的 `RUL` 定義仍然是：

```text
rul_hat = health_remaining_hat * total_life
```

這裡要明確保留邊界：

- 這個定義只在 benchmark 評估內有效
- 它還不是 simulator 端的最終推理規則

## 4. 目前的預設設定

- 預設 active model：
  - `LSTM`
- Families：
  - `A2,A3,A4`
- Grouped evaluation：
  - `5-fold`
- Window length：
  - `32`
- Batch size：
  - `64`
- Fine-tuning epochs：
  - `100`
- Fine-tuning patience：
  - `10`
- Pretraining epochs：
  - `20`
- Pretraining patience：
  - `5`
- Learning rate：
  - `1e-3`
- Weight decay：
  - `1e-5`
- `A2` long-life multiplier：
  - `2.0`
- Monotone projection：
  - enabled
- Family tuning：
  - 在 `LSTM`-only run 中啟用

## 5. 本批次明確沒有做的事情

- 沒有改動 `run_experiment.py`
- 沒有改動 RL 主流程
- 沒有新增 raw `RUL` 直接回歸 head
- 沒有加入 unified conditional model
- 沒有做 feed interpolation
- 沒有把 particle-size curve 加進輸入特徵
- 沒有重設 simulator 端 inference 規則
- 沒有刪除 `GRU / TCN / ATTENTION`

## 6. 執行方式

執行預設的優化版 `LSTM` run：

```bash
python benchmark_sequence_models.py
```

執行四模型 compare：

```bash
python benchmark_sequence_models.py --models GRU,LSTM,TCN,ATTENTION
```

關閉 family tuning，快速跑單模型：

```bash
python benchmark_sequence_models.py --disable-family-tuning
```

## 7. 如何解讀輸出

本批次 benchmark 在以下條件下才算真正成功：

- `A3` 相對鎖定基線有提升
- `A2` 沒有退化
- `A4` 沒有退化
- projected `RUL` 比舊版 slope-based benchmark 更平順
- compare artifacts 能清楚說明 tuned pass 是否應取代 shared-config pass

## 8. 本批次之後的下一步

當完整優化 run 跑完後，下一步應該是：

- 檢查新的 baseline-compare artifacts
- 判斷 tuned `LSTM` family configs 是否應取代目前參考設定
- 只有在這一步確認之後，才考慮把 benchmark 輸出重新接回 simulator 相關工作
