# Iteration Changelog And TODO

## 1. 本輪目標

本輪迭代對應 `POMCP_DQN_result_root_cause_analysis.md` 中的三個主要方向：

- 修正 `DQN` 的 train/eval route 不一致問題，避免 constrained-only 訓練直接作為 unrestricted 官方結果
- 對齊 maintenance reward 與 final eval metric 的成本尺度
- 對 `POMCP` 的 generative model 做最小對齊，讓 rollout 退化與 Region-B elapsed 更接近真實 env

另外，本輪新增一套 `maint_only compare`，用固定 `DQN` scheduler 來隔離 maintenance policy 差異。

## 2. 已完成改動

### 2.1 Route-aware 訓練 / 評估

- 在 `config.py` 新增：
  - `TRAIN_POLICY_ROUTES = ("region_on", "region_off")`
  - `ENABLE_OOD_DIAGNOSTIC_EVAL = False`
- `run_experiment.py` 現在會依序跑：
  - `seed`
  - `train_policy_route`
  - `maint_mode`
- `train_one_mode(...)` 新增：
  - `train_enforce_region`
  - `train_policy_tag`
- official periodic eval / final eval 改成 same-route only
- dual-route OOD eval 改成可選診斷模式，預設不啟用

### 2.2 Compare 流程擴充

- 保留 `full_system compare`
- 新增 `maint_only compare`
  - anchor 固定為同 route 下的 `DQN` scheduler
  - 比較：
    - `DQN scheduler + DQN maintenance`
    - `同一 DQN scheduler + POMCP maintenance`
- 新增 route difference 輸出：
  - `route_compare_full_system_<maint_mode_tag>`
  - `route_compare_maint_only_anchor_dqn_<maint_mode_tag>`
- compare summary 現在帶：
  - `compare_type`
  - `train_policy_tag`
  - `eval_policy_tag`
  - `scheduler_anchor`

### 2.3 Reward 對齊

- `maintenance_reward(...)` 已改為與 `compute_costs()` 同量級的 duration-based maintenance cost
- 新 reward 現在使用：
  - `downtime_cost = dur * local_urgency`
  - `maint_cost = IM_COST * dur` 或 `CM_COST * dur`
  - `dn_risk_cost = W_RISK * risk_t` only for `DN`
  - `window_violation` penalty
- `MAT_COST_*` 保留給 logging / accounting，但不再是 maintenance RL 主 reward
- `POMCP` rollout reward 與 `DQN` 現在共用同一公式

### 2.4 POMCP generative model 最小對齊

- 在 `src/env.py` 新增：
  - `rul_from_operating_index(...)`
  - `operating_index_from_rul(...)`
- `generative_step(DN)` 不再使用固定 `POMCP_H_DECAY`
- 現在 rollout 退化改為：
  - `expected_pt = _mean_proc_time(mid)`
  - `effective_rate = BASE_DEGRADATION_RATE * (1 + DEGRAD_ALPHA * stress)`
  - `delta_idx = effective_rate * expected_pt / PT_REF`
  - 再透過 RUL / operating-index 映射回 `h`
- `region_b_elapsed` 在 rollout 中改為累加 `expected_pt`
- `IM` / `CM` 的 baseline 語義在真實 env 與 generative model 保持一致

## 3. 受影響文件與關鍵接口

主要代碼改動集中在：

- `config.py`
- `run_experiment.py`
- `src/env.py`
- `src/compare.py`

關鍵接口變化：

- `train_one_mode(...)`
  - 新增 `train_enforce_region`
  - 新增 `train_policy_tag`
- compare summary schema
  - 新增 `compare_type`
  - 新增 `train_policy_tag`
  - 新增 `eval_policy_tag`
  - 新增 `scheduler_anchor`
- env helper
  - 新增 `rul_from_operating_index(...)`
  - 新增 `operating_index_from_rul(...)`

## 4. 行為變化摘要

### 官方結果語義改變

- 以前：單一訓練 run 會同時輸出 constrained / unrestricted final eval
- 現在：官方結果只輸出 same-route eval
  - `region_on` train -> `region_on` official eval
  - `region_off` train -> `region_off` official eval

### Compare 語義變更

- `full_system compare`
  - 仍然比較完整策略組合
- `maint_only compare`
  - 用同一個 `DQN` scheduler，只替換 maintenance policy
  - 可以直接看 maintenance 層的策略差異

### POMCP rollout 語義變更

- 以前：DN rollout 只按固定 `POMCP_H_DECAY` 減 `h`
- 現在：DN rollout 會參考 `BASE_DEGRADATION_RATE`、`PT_REF`、`_mean_proc_time(mid)` 與 stress

## 5. 驗證結果

### 5.1 靜態檢查

已通過：

```bash
python -m py_compile config.py run_experiment.py infer_demo.py checkpointing.py src/env.py src/compare.py src/viz.py
```

### 5.2 Route-aware / compare smoke test

在 sandbox 內跑 Python 會遇到 OpenMP `SHM2` 問題，因此改用 sandbox 外做極小 smoke test。

驗證內容：

- `region_on + DQN`
- `region_on + POMCP`
- `region_off + DQN`
- `region_off + POMCP`
- `maint_only compare`
- `full_system compare`

結果：

- 四個 route/mode 組合都能完成 1-episode 訓練 + official eval
- `evaluate_maint_only_results(...)` 可正常產生 `DQN` / `POMCP` 兩臂結果
- `compare_mode_results(...)` 可正常產生 `full_system` compare summary

煙霧測試輸出摘要：

- `region_on_hx0p3_hy0p1`
  - `DQN official total = 0.0`
  - `POMCP official total = 0.0`
- `region_off_unrestricted`
  - `DQN official total = 401.935...`
  - `POMCP official total = 550.0`

這個 smoke 只用來驗證流程與輸出結構，不用來判斷策略優劣。

## 6. 已知限制

- 本輪官方配置仍是單 seed：`EXPERIMENT_SEEDS = (42,)`
- `maint_only compare` 目前只支持 `DQN` scheduler anchor
- `route_compare_maint_only_anchor_dqn_*` 比較的是兩條 route 各自的 `DQN` anchor scheduler，不是同一個 scheduler 跨 route 共用
- `POMCP` 目前仍不是高保真 queue-aware rollout，只是從固定 `h` 衰減提升到與真實 env 更接近的 mean-pt / index-space 衰減
- `infer_demo.py` 尚未擴成 route-aware 主入口；目前主交付集中在 `run_experiment.py`

## 7. TODO List

### P0

- 用正式 episode 數重新跑 route-aware 實驗，確認：
  - `DQN region_off` 不再是 constrained-only OOD 結果
  - `full_system compare` 與 `maint_only compare` 輸出都穩定
- 核查新的 `DQN unrestricted` 是否仍出現高 RUL 下大量 `CM`

### P1

- 為 route-aware runner 補一份聚合報表：
  - 依 `train_policy_tag`
  - 依 `compare_type`
  - 依 `maint_mode`
- 把 `infer_demo.py` 同步到新的 route-aware compare schema
- 檢查是否要把 `MAT_COST_*` 從 maintenance 決策 log 中標記為 legacy，避免後續分析混淆

### P2

- 把 `POMCP` rollout 從 mean-pt 近似升級為局部 queue-aware 模型
- 研究是否要增加第二個 `scheduler anchor`：
  - `POMCP scheduler anchor`
  - 或雙 anchor compare
- 擴展為多 seed 正式實驗，重新檢查：
  - route-aware 公平性
  - `DQN vs POMCP` 的穩定差異

