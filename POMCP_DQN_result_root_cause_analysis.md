# POMCP vs DQN 結果根因分析

## 1. 結果總結

本分析基於最新 paired output：`outputs/paired_20260310_083048`。

核心觀察如下：

| Policy route | DQN | POMCP | 結論 |
| --- | --- | --- | --- |
| `region_on_hx0p3_hy0p1` | `tard=177.21`, `maint=597.20`, `total=774.41`, `makespan=3770.46` | `tard=159.37`, `maint=836.45`, `total=995.82`, `makespan=3821.75` | `POMCP` 稍微降低 tardiness，但 maintenance 成本更高，總成本反而更差 |
| `region_off_unrestricted` | `tard=684.41`, `maint=24200.00`, `total=24884.41`, `makespan=3880.14` | `tard=101.72`, `maint=329.90`, `total=431.62`, `makespan=3770.46` | `DQN` 明顯崩潰，`POMCP` 顯著更穩定 |

結果文件：

- `outputs/paired_20260310_083048/aggregate_compare_region_on_hx0p3_hy0p1.json`
- `outputs/paired_20260310_083048/aggregate_compare_region_off_unrestricted.json`
- `outputs/paired_20260310_083048/seed_0042/compare/maint_compare_20260310_083048_region_on_hx0p3_hy0p1.json`
- `outputs/paired_20260310_083048/seed_0042/compare/maint_compare_20260310_083048_region_off_unrestricted.json`

額外要點：

- constrained 下，兩者 dispatch 次數都為 `279`，但 `POMCP` 的 makespan 仍多出約 `51.29`。
- unrestricted 下，差異不是小幅偏移，而是策略型態完全不同：
  - `DQN`: `DN=0`, `IM=0`, `CM=242`
  - `POMCP`: `DN=266`, `IM=13`, `CM=0`
- unrestricted compare 的 divergence 是 `281 / 281`，說明兩條策略幾乎完全分岔。

## 2. 直接證據

### 2.1 Constrained 下的差異

`outputs/paired_20260310_083048/seed_0042/compare/maint_compare_20260310_083048_region_on_hx0p3_hy0p1.json` 顯示：

- `DQN` 動作分佈：`DN=266`, `IM=5`, `CM=5`
- `POMCP` 動作分佈：`DN=270`, `IM=1`, `CM=8`
- `divergence_count=41`, `decision_union_count=289`

這表示 constrained 規則已經把兩者策略空間壓得很窄，兩個方法沒有走向完全不同的行為模式；`POMCP` 只是比 `DQN` 更保守，付出更多 maintenance cost。

### 2.2 Unrestricted 下的異常

`outputs/paired_20260310_083048/seed_0042/compare/maint_compare_20260310_083048_region_off_unrestricted.json` 顯示：

- `DQN` 動作分佈：`DN=0`, `IM=0`, `CM=242`
- `POMCP` 動作分佈：`DN=266`, `IM=13`, `CM=0`
- `divergence_count=281`, `decision_union_count=281`

更關鍵的是，`DQN` 在高 RUL 區間就大量選 `CM`。compare rows 中可見多個例子：

- `mid=0`, `maint_seq_machine=1`, `DQN=CM`, `h=0.9749`, `dur=20.0`
- `mid=0`, `maint_seq_machine=2`, `DQN=CM`, `h=0.9799`, `dur=20.0`
- `mid=0`, `maint_seq_machine=3`, `DQN=CM`, `h=1.0000`, `dur=20.0`
- `mid=0`, `maint_seq_machine=4`, `DQN=CM`, `h=0.9738`, `dur=20.0`

這不是單點噪聲，而是明顯的 policy collapse。

## 3. 主要原因與代碼定位

### 3.1 `DQN unrestricted` 目前是 OOD 評估

現象：

- `DQN` 訓練時只看過 constrained 規則，但 final eval 同時測 constrained 與 unrestricted。
- 因此 unrestricted 結果不是「同分布測試」，而是 out-of-distribution evaluation。

代碼依據：

- `train_one_mode()` 一開始固定 `cfg.ENFORCE_REGION_POLICY = True`：`run_experiment.py:1147`
- periodic eval 也固定 `eval_cfg.ENFORCE_REGION_POLICY = True`：`run_experiment.py:1507`
- 只有 final eval 才在 `for enforce_region in (True, False)` 中雙路測試：`run_experiment.py:1582`

影響判斷：

- `DQN unrestricted` 的崩潰有很大機率是分布偏移造成，不應直接解讀為「DQN 天生不如 POMCP」。

修正方向：

- 如果要比較 unrestricted，`DQN` 也必須在 unrestricted 或 mixed-policy 條件下訓練。

### 3.2 Region action mask 讓 `DQN` 在 unrestricted 下暴露未學過的動作值

現象：

- constrained 訓練時，Region A 只能 `DN`，Region C 只能 `CM`。
- 這會導致大量 state-action pair 根本沒被學過。
- unrestricted 評估時 mask 移除，未受約束的 `Q(IM)` / `Q(CM)` 直接進入 argmax，容易出現失真行為。

代碼依據：

- Region mask / enforce 規則：`run_experiment.py:90`、`run_experiment.py:99`
- `DQN` 選動作時直接吃 `allowed_actions`：`run_experiment.py:1009`
- `MaintenanceAgentDDQN.act()` 在 exploration 與 exploitation 兩路都受 `allowed_actions` 限制：`src/agents.py:33`

影響判斷：

- 這能直接解釋為什麼 unrestricted compare 中，`DQN` 在 `h≈0.97~1.00` 的狀態仍反覆選 `CM`。
- constrained 訓練中，那些高 RUL 狀態下的 `CM` 值並沒有被穩定學習或校正；一旦 mask 拿掉，錯誤 Q 值就可能主導行為。

修正方向：

- 不要用 constrained-only 訓練去直接評估 unrestricted。
- 若仍要跨域評估，至少加入 action uncertainty handling 或 fallback rule，避免未覆蓋動作裸露參與 argmax。

### 3.3 訓練 reward 與最終評估指標不一致，`CM` 在訓練端被低估

現象：

- maintenance agent 的訓練 reward 與 final eval / summary 的 maintenance 成本不是同一套尺度。
- 對 `CM` 特別明顯。

代碼依據：

- 訓練 reward：`maintenance_reward()` 使用 `dur * local_urgency + material_cost + risk + violation`：`run_experiment.py:116`
- `local_urgency` 會被截斷到 `1.0`：`config.py:103`、`src/env.py:531`
- 訓練端材料成本：`MAT_COST_CM = 2.0`：`config.py:108`
- summary / final eval 成本：`CM_COST * duration`、`IM_COST * duration`：`src/env.py:895`
- 其中 `CM_COST = 5.0`、`IM_COST = 1.0`：`config.py:148`

具體量級：

- 目前 `CM` duration 固定是 `20.0`：`config.py:77`
- 訓練端對一次 `CM` 的主要時間項最多約是 `20 * local_urgency <= 20`
- 再加上 `MAT_COST_CM=2` 與 `risk<=1`，單次 `CM` reward 對應成本大致在 `22~23` 量級
- 但 final eval 端對同一次 `CM` 的 maintenance 成本是 `CM_COST * 20 = 100`

影響判斷：

- `CM` 在訓練端被明顯低估，足以誘發過度維護。
- 這個問題會影響 `DQN` 與 `POMCP`，但 `DQN` 直接學 Q 值，對 reward 尺度偏差更敏感。

修正方向：

- 對齊 maintenance 訓練 reward 與 summary 成本尺度。
- 最低限度要讓 `CM` / `IM` 的 duration penalty 在訓練端與評估端具有一致量級。

### 3.4 `POMCP` 的規劃模型與真實環境存在失配

現象：

- `POMCP` 在 constrained 下能稍微降低 tardiness，但 maintenance 成本更高、makespan 更差。
- 這更像是 planner 在優化一個簡化模型，而不是真實系統。

代碼依據：

- `POMCP` 規劃時使用的 generative model 來自 `env.generative_step()`：`run_experiment.py:1036`
- `generative_step()` 只做簡化的 health transition：
  - `CM` / `IM` 直接改寫健康狀態：`src/env.py:657`
  - 退化只用 `POMCP_H_DECAY` 做固定步長衰減：`src/env.py:673`
  - `region_b_elapsed` 在 `DN` 下用 `OBS_PROC_DEFAULT` 近似累加：`src/env.py:676`
- 這個模型沒有顯式模擬：
  - queue 演化
  - dispatch rule 的後果
  - 真實工序時間對健康與 tardiness 的鏈式影響

影響判斷：

- 這可以解釋 `POMCP` 為什麼在 constrained 下傾向多做維護，以換取較低的風險與 tardiness，但最終總成本反而更高。
- 還要注意：`pf_dn/pf_im/pf_cm` 特徵本身也是由 `estimate_p_fail_horizon()` 基於同一 generative model 算出來的：`run_experiment.py:1308`。因此模型失配不只影響 `POMCP` 規劃，也會污染 `DQN` 的狀態特徵，只是 `POMCP` 受影響更直接。

修正方向：

- 若要把 `POMCP` 結果當真，需要先改善 `generative_step()`，至少讓其更接近真實 dispatch / queue / processing-time 造成的後果。

### 3.5 當前 paired runner 比較的是完整策略組合，不是 maintenance-only

現象：

- 當前 `DQN vs POMCP` 比較，不只是 maintenance policy 不同，scheduler 也是各自獨立訓練。

代碼依據：

- 每個 mode 都獨立調用 `train_one_mode()`：`run_experiment.py:1710`
- `train_one_mode()` 內部每次都重新建立自己的 `THDQN scheduler`：`run_experiment.py:1154`

影響判斷：

- `DQN vs POMCP` 的差異同時混入 scheduler policy 變化。
- 因此，不能把所有 `tard`、`makespan` 或 even maintenance timing 差異都歸咎於 maintenance planner。

修正方向：

- 如果目標是純 maintenance policy 比較，需固定 scheduler，只替換 maintenance layer。

### 3.6 「dispatch 重複取 noisy RUL」不是這次結果的主要原因

現象：

- review finding 提到 `dispatch()` 可能重複使用 noisy RUL，導致 log 與 breakdown probability 不一致。

代碼依據：

- 目前 `dispatch()` 僅調用一次 `_peek_rul(mid)`，並將同一個 `h` 用於 log 與 breakdown probability：`src/env.py:758`

影響判斷：

- 這個問題在目前版本不是主要嫌疑，無法解釋 unrestricted 下 `DQN` 的全面 `CM` collapse。

修正方向：

- 可以保留為 code hygiene 檢查項，但不應視為本輪結果分岔的主因。

## 4. 優先修正項

### P0

- 若要比較 unrestricted，必須新增：
  - `DQN constrained train -> constrained eval`
  - `DQN unrestricted train -> unrestricted eval`
  - 或 `DQN mixed train -> dual eval`

### P1

- 對齊 maintenance 訓練 reward 與 final eval cost，特別是 `CM` / `IM` 的 duration cost。
- 在跨域評估前，處理 constrained-training 下未覆蓋 action 的 fallback / uncertainty。

### P2

- 改善 `src/env.py:645` 的 `generative_step()`，讓 `POMCP` 用到的模型更接近真實環境。
- 若目標是只比較 maintenance，固定 scheduler，不讓兩個 mode 各自訓練不同 scheduler。

## 5. 建議驗證實驗

1. 重新訓練 `DQN unrestricted`，只測 unrestricted，檢查是否仍出現 `DN=0, IM=0, CM=242`。
2. 重新訓練 `DQN mixed-policy`，同時測 constrained 與 unrestricted，檢查 dual-route 泛化是否改善。
3. 對齊 reward 後再跑一次 paired compare，觀察 `DQN unrestricted` 是否還會在 `h≈1.0` 選 `CM`。
4. 固定同一個 scheduler，只替換 maintenance mode，再比較 `POMCP vs DQN`。
5. 對比 `src/env.py:645` 的 `generative_step()` 預測軌跡與真實 env 內 dispatch 後的 health 軌跡，量化模型失配。

## 6. 結論

目前這組結果最可信的解讀是：

- constrained 下，`DQN` 更實用；`POMCP` 雖然稍降 tardiness，但 maintenance 過重。
- unrestricted 下，`DQN` 的結果目前不能當成公平比較，因為它實際上是在 constrained-only 訓練後做 OOD 測試。
- `POMCP` 的表現雖然在 unrestricted 下更穩，但其規劃模型仍有明顯失配，不能直接視為最終可信最優解。

因此，當前最需要先修的不是某個單點 bug，而是實驗設計本身：

- 先讓 train/eval policy route 對齊
- 再對齊 training reward 與 final metric
- 最後才去評估 `POMCP` 與 `DQN` 的真實優劣

補充限制：

- 目前 `EXPERIMENT_SEEDS = (42,)`，只有單一 seed：`config.py:199`
- 因此 `aggregate_compare_*.json` 中的 `std=0.0` 不是穩定收斂，而只是目前樣本數為 1

