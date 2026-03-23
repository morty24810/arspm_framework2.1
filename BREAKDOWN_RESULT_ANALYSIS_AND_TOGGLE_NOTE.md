# Breakdown Result Analysis And Toggle Note

## 本輪結果摘要

基於 `outputs/paired_20260311_103106/paired_eval_rows.json` 與 `outputs/paired_20260311_103106/paired_compare_rows.json`，目前可以先得到幾個明確結論：

- 這一輪改動是有效的：`DQN unrestricted` 已經不再退化成「全 DN / 零維護」。
- 現在結果的主導因素已經不是單純的 `IM/CM` 選擇，而是 `breakdown` 次數與其直接成本。
- `POMCP` 在目前配置下，無論 constrained 或 unrestricted，都落後於 `DQN`。
- `maint_only compare` 現在已經有真實 decision log，因此可以比較有把握地說：目前主要問題在 maintenance policy 本身，而不只是 scheduler。

## 直接觀察

### Constrained full-system

- `DQN`: `total=6066.05`, `breakdown=4`, `breakdown_cost=5168`
- `POMCP`: `total=9640.69`, `breakdown=7`, `breakdown_cost=9044`

解讀：

- `POMCP` 的 tardiness 較低，但 breakdown 更多，maintenance / failure 成本顯著更高。
- 目前 constrained 下的勝負幾乎是由 breakdown 次數決定。

### Unrestricted full-system

- `DQN`: `total=2644.41`, `breakdown=1`, `breakdown_cost=1292`
- `POMCP`: `total=8065.33`, `breakdown=4`, `breakdown_cost=5168`

解讀：

- `DQN` 在 unrestricted 下明顯更穩。
- `POMCP` 雖然做了更多維護，但沒有有效避免 breakdown，導致總成本仍然偏高。

### Maint-only compare

#### Constrained maint-only

- `DQN`: `total=1765.39`, `breakdown=1`
- `POMCP`: `total=9922.07`, `breakdown=7`

#### Unrestricted maint-only

- `DQN`: `total=1679.92`, `breakdown=0`
- `POMCP`: `total=12090.40`, `breakdown=7`

解讀：

- 在 scheduler 固定為同一個 `DQN scheduler` 的前提下，單換 maintenance policy，`POMCP` 仍然更差。
- 因此這一輪結果可以初步判斷：主要差異來自 maintenance planner，而不只是 scheduler 差異。

## 目前結果的主要解讀

### 1. DQN 的 reward 耦合修正是有效的

這一輪 `DQN unrestricted` 已不再是上一版那種「完全不維護」的極端策略。

- `unrestricted full-system DQN`: `DN=245, IM=24, CM=9`
- `unrestricted maint-only DQN`: `DN=234, IM=31, CM=11`

這表示：

- 開啟真實 breakdown
- 把 expected breakdown loss 轉進 maintenance reward

這兩件事確實改變了 `DQN` 的行為方向。

### 2. POMCP 的問題不是單純「怕壞」，而是維護時機不準

- constrained 下，`POMCP` 並沒有大量維護，但 breakdown 仍然偏多。
- unrestricted 下，`POMCP` 又變成大量 `IM/CM`，但仍無法有效壓低 breakdown。

這更像是：

- 該修時沒有修到位
- 不該修時又修太多

也就是 rollout 對最佳維護時點的建模仍然不夠準。

### 3. Hx/Hy 現在更像是限制而不是保護

目前結果顯示：

- `DQN full-system`: unrestricted 比 constrained 更好
- `POMCP full-system`: unrestricted 也比 constrained 更好

這代表在目前 breakdown 語義下，`Hx/Hy` 這組人為先驗更像是在壓縮本來可以提早介入的決策空間。

## 方法論上的保留

目前比較仍不是完全乾淨的 paired compare。

原因是 `DQN` 和 `POMCP` 在 compare 流程中，雖然共用同一份 scenario bank，但 `env_breakdown` 的隨機流仍然是分開取 seed 的。這表示：

- scenario 是對齊的
- 但 breakdown randomness 還沒有完全對齊

因此目前這批結果：

- 趨勢可信
- 但差值幅度仍然混有 stochastic breakdown 的噪聲

## 我希望增加的設計開關

我希望增加一個明確的開關，用來**關閉隨機 breakdown**，只保留 deterministic / threshold-based 的故障語義，理由如下：

- 一般機器未必會如此頻繁地隨機故障。
- 目前 stochastic breakdown 的代價過於劇烈，尤其是：
  - breakdown recovery time 很長
  - breakdown 直接拉高 total cost
  - 容易讓策略學習被單次隨機事件主導
- 如果目標是先分析 maintenance / scheduling 策略本身，過強的隨機故障會讓比較結果變得不夠穩定。

### 我希望的開關語義

建議新增一個顯式配置，例如：

- `ENABLE_STOCHASTIC_BREAKDOWN = True / False`

建議語義：

- `True`
  - 保留目前 stochastic breakdown
  - 同時保留 `true RUL <= HARD_BREAKDOWN_RUL` 的硬故障
- `False`
  - 關閉 stochastic breakdown
  - 只保留硬閾值故障，也就是 `true RUL <= HARD_BREAKDOWN_RUL` 時才 breakdown

這樣可以把實驗拆成兩種模式：

1. `hard-threshold only`
   - 用來觀察策略在較穩定、較可解釋的故障機制下的行為。
2. `hybrid breakdown`
   - 用來觀察策略在帶有隨機風險時的表現。

## 目前的工程判斷

如果下一輪要先追求可解釋性與穩定對比，優先順序應該是：

1. 先加入 `關閉 stochastic breakdown` 的開關。
2. 先跑一輪 `hard-threshold only` 的結果。
3. 再和 `hybrid breakdown` 結果做對比。

這樣才能分清楚：

- 問題是 maintenance policy 本身
- 還是 stochastic breakdown 的懲罰設計太強，掩蓋了策略真實差異
