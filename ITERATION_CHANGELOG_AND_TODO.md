# Iteration Changelog And TODO

## 1. 本輪目標

本輪迭代把重點從「route-aware compare 公平性」往前推到真正的故障耦合：

- 讓 breakdown 在官方實驗裡真正發生，而不是只停留在抽象風險
- 把 breakdown 帶來的延誤 / 重派 / 物理成本轉進 maintenance reward
- 保留雙 agent 架構，但補上最低限度的 scheduling-side failure awareness
- 修掉 `maint_only compare` 沒有 decision log 的問題

## 2. 已完成改動

### 2.1 真實 Breakdown 已啟用

- `config.py`
  - `BREAKDOWN_ENABLE = True`
  - 新增 `HARD_BREAKDOWN_RUL = 0.05`
  - 新增 `BREAKDOWN_REQUEUE_MODE = "restart_op"`
  - 新增 `SCHED_SAFE_DISPATCH = True`
- `src/env.py`
  - 新增 `peek_rul_true(mid)`，把 breakdown physics 從 noisy observation 分離出來
  - `dispatch()` 現在先預估本次加工的真實退化，再決定：
    - 是否觸發 hard breakdown
    - 是否觸發 stochastic breakdown
  - 一旦 breakdown：
    - 機台立刻進入 `MAINT`
    - recovery duration = `MT_CM + FAIL_EXTRA_DUR`
    - 工序中斷並退回等待重派
    - machine baseline / true RUL recovery 後重置為 `1.0`

### 2.2 Breakdown 成本與狀態追蹤補齊

- `src/env.py` 新增 breakdown accounting：
  - `breakdown_count`
  - `hard_breakdown_count`
  - `stochastic_breakdown_count`
  - `breakdown_cost_total`
  - `requeued_op_count`
  - `interrupted_proc_time`
- `compute_costs()` 中 `BREAKDOWN` 不再等同普通 `CM`
  - 現在使用：
    - `FAIL_COST_MULT * recovery_dur`
    - `SCRAP_PART_COST`
    - `FAIL_PENALTY`
- `timeline_ops` 改成可記錄 segment status：
  - `DONE`
  - `INTERRUPTED`

### 2.3 Maintenance reward 改成吃 expected breakdown loss

- `run_experiment.py`
  - `maintenance_reward(...)` 現在語義是：
    - `DN -> expected_breakdown_loss + violation`
    - `IM/CM -> downtime_cost + maintenance_cost + violation`
  - 不再用 `W_RISK * risk_t` 作為主 reward
- 新增 `compute_breakdown_reward_terms(...)`
  - 用 env 的 breakdown 預估資訊算：
    - `p_fail_exec`
    - `expected_breakdown_loss`
    - `expected_redispatch_pt`
    - `breakdown_recovery_dur`
- `DQN` 訓練與 `POMCP` rollout reward 現在共用這套公式

### 2.4 Scheduling 端補上最低限度耦合

- scheduler state 從 `12 -> 14`
- 新增兩個特徵：
  - `idle_fail_risk_mean`
  - `idle_fail_risk_max`
- `dispatch()` 新增 safe-dispatch filter：
  - 若同一工序存在安全 idle machine，則只在安全 machine 內套用原本 rule
  - 只有全部候選 machine 都不安全時，才允許派到高風險 machine

### 2.5 POMCP 生成模型對齊 breakdown

- `src/env.py::generative_step(...)`
  - `DN` 現在會模擬：
    - hard breakdown
    - stochastic breakdown
    - forced recovery transition
  - breakdown 分支會回到 recovery 後狀態：
    - `baseline_rul = 1.0`
    - `h = 1.0`
    - `kind = "BREAKDOWN"`
    - `dur = breakdown_recovery_duration`

### 2.6 Compare / 輸出修正

- `evaluate_once(...)` 新增 `collect_decision_log`
  - `maint_only compare` 現在會真的帶回 decision log
- `decision_log` / summary / compare 補上 breakdown 欄位：
  - `breakdown_count`
  - `breakdown_cost`
  - `requeued_op_count`
  - `interrupted_proc_time`
  - `hard_breakdown_count`
  - `stochastic_breakdown_count`
  - `breakdown_kind`
- `src/viz.py`
  - Gantt 現在會把 interrupted 工序段畫出來
  - RUL 曲線會標出 `BREAKDOWN` 時刻
  - compare 圖會展示 breakdown / requeue 摘要

## 3. 受影響文件與關鍵接口

主要改動集中在：

- `config.py`
- `src/env.py`
- `run_experiment.py`
- `src/compare.py`
- `src/viz.py`

關鍵接口變化：

- `evaluate_once(...)`
  - 新增 `collect_decision_log`
- scheduler state
  - `THDQNAgent.state_dim: 12 -> 14`
- env helper
  - 新增 `peek_rul_true(...)`
  - 新增 `processing_delta_idx(...)`
  - 新增 `expected_breakdown_loss(...)`
  - 新增 `is_safe_dispatch(...)`

## 4. 行為變化摘要

### 4.1 `DN` 不再天然便宜

- 以前：
  - `BREAKDOWN_ENABLE=False`
  - unrestricted 下 `DN` 容易退化成幾乎零維護成本策略
- 現在：
  - `DN` 會顯式吃到預期 breakdown loss
  - 真實 env 也會發生 mid-process breakdown

### 4.2 breakdown 會改變排程而不是只改 maintenance 成本

- 故障中的機台不可分派
- 被中斷的工序會回到待重派狀態
- tardiness 現在會透過真實重派與停機時間累積，而不只是 maintenance proxy

### 4.3 `maint_only compare` 不再是空結果

- 以前：
  - `generate_outputs=False` 時 `decision_log=None`
  - compare 只能得到全零 action counts
- 現在：
  - `collect_decision_log=True` 時可保留 decision log
  - `maint_only compare` 能產生真實行為差異

## 5. 驗證結果

### 5.1 靜態檢查

已通過：

```bash
python -m py_compile config.py run_experiment.py infer_demo.py checkpointing.py src/env.py src/compare.py src/viz.py
```

### 5.2 Breakdown / safe-dispatch smoke test

在 sandbox 外做了小型 Python smoke test，驗證：

- 不安全機台存在時，safe-dispatch 會選擇安全機台
- `true RUL <= 0.05` 時會立即 hard breakdown
- breakdown 後：
  - machine 進入 `MAINT`
  - `busy_until = t_fail + MT_CM + FAIL_EXTRA_DUR`
  - job op 不完成
  - `interrupted_count` 增加
  - `timeline_ops` 記錄 `INTERRUPTED`

驗證輸出摘要：

- `safe_m0=False, safe_m1=True` 時，dispatch 選到了 `mid=1`
- 強制 hard breakdown 時：
  - `breakdown_count=1`
  - `hard_breakdown=True`
  - `requeued=True`
  - `machine0_status=MAINT`
  - `timeline_last=(..., 'INTERRUPTED')`

### 5.3 Decision-log collection smoke test

在 sandbox 外跑了一次極小 `evaluate_once(...)`：

- `generate_outputs=False`
- `collect_decision_log=True`

結果：

- `decision_log_len = 98`
- `sched_events = 49`
- `maint_events = 49`

說明 `maint_only compare` 所需的 decision log 已能正常收集。

## 6. 已知限制

- 目前 `compute_breakdown_reward_terms(...)` 用的是當前 env / slack 驅動的近似執行風險，不是完整 queue-aware expected loss
- hard breakdown 使用固定閾值 `0.05`，尚未做靈敏度實驗
- breakdown 後工序採整道重做，暫未支持 partial resume
- safe-dispatch 只在「已選定工序的 machine 選擇」上生效，尚未把 safety 反向傳回工序選擇層
- `infer_demo.py` 沒有擴成新的 breakdown-focused分析入口；主交付仍在 `run_experiment.py`

## 7. TODO List

### P0

- 用正式 episode 數重跑：
  - `region_on + DQN`
  - `region_on + POMCP`
  - `region_off + DQN`
  - `region_off + POMCP`
- 核查 unrestricted 下：
  - `DQN` 是否仍偏向過度 `DN`
  - `POMCP` 是否仍偏向過度 `IM`
- 直接查看新的 `maint_only compare`，確認雙方差異是否主要來自 maintenance 而不是 scheduler

### P1

- 把 breakdown event 的更多細節寫進 compare row：
  - `jid`
  - `oid`
  - `t_fail`
  - `recovery_end`
- 為 Gantt 加上更明確的 interrupted / breakdown 圖例
- 對 hard-threshold `0.05` 做 sweep，檢查策略敏感度
- 檢查 safe-dispatch 是否需要升級為「跨工序選擇也考慮 safety」

### P2

- 把 expected breakdown loss 從 mean-pt 近似升級為 queue-aware / job-aware 近似
- 研究是否要讓 scheduling reward 也顯式吃到 breakdown penalty transfer，而不只透過 env 結果間接感知
- 若未來還要更強耦合，再評估：
  - centralized critic
  - shared reward
  - joint maintenance-scheduling planner
