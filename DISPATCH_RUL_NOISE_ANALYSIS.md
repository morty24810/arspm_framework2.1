# Dispatch RUL Noise Analysis

## Finding

Review finding:

> `dispatch` calls `_peek_rul` twice; with observation noise this can log one value and compute breakdown probability with another, adding unintended randomness.

## Current Code Status

這個問題在目前版本裡，**原始描述已不成立**。

證據：

- `dispatch()` 目前只取一次 `h`：`src/env.py:789`
- 同一個 `h` 同時用於：
  - 記錄 RUL：`src/env.py:790`
  - 計算 breakdown probability：`src/env.py:792`

也就是說，現在不存在「同一次 dispatch 內因為重複 `_peek_rul()` 導致兩個不同 noisy RUL」的情況。

## Residual Risk

雖然「重複取值」已經不是問題，但仍有一個**剩餘風險**：

- `_peek_rul()` 目前回傳的是 **帶 observation noise 的 RUL**：`src/env.py:536`
- 其內部走的是 `GRUCache.get_h_obs(...)`：`src/env.py:538`
- `get_h_obs(...)` 會在 `get_h(...)` 的基礎上加噪聲：`src/sensor_bank.py:54`

因此目前的 breakdown physics 實際上是：

- 用**觀測值** `h_obs` 來算 `p_break`
- 而不是用**真實值** `h_true`

## Why This Still Matters

這代表目前 `dispatch()` 雖然沒有「一個 dispatch 取兩次不同 noisy RUL」的 bug，但仍然存在以下建模問題：

1. `breakdown probability` 會被 observation noise 直接擾動  
   同一個真實 machine state，只因為 sensor noise 不同，就可能得到不同 `p_break`。

2. 物理狀態與觀測狀態沒有分離  
   如果設計目標是「觀測有噪聲，但故障機理由真實退化決定」，那現在的寫法會把 sensor noise 混進 physics。

3. 評估結果可能增加非必要隨機性  
   這種隨機性不是來自 degradation 或 dispatch 本身，而是來自 noisy observation 直接進入 breakdown model。

## Recommended Fix

如果你要保留「觀測噪聲」但避免它污染物理故障機制，推薦做法是：

- log / decision 顯示用 `h_obs`
- breakdown physics 用 `h_true`

具體上可行的最小修正是：

- 在 `EventDrivenShopEnv` 增加一個 noise-free accessor，例如：
  - 直接走 `GRUCache.get_h(...)`
  - 或新增 `_peek_rul_true(mid)`
- `dispatch()` 裡：
  - `h_obs` 用於 `_log_rul(...)`
  - `h_true` 用於 `failure_prob(h_true)` 與 `p_break`

## Conclusion

結論分兩層：

- 原 review finding 的字面問題：**已基本修掉**
  - 同一次 `dispatch()` 現在只取一次 `_peek_rul()`
- 但更深一層的問題仍然存在：
  - **breakdown physics 仍使用 noisy RUL，而不是 true RUL**

所以這條 finding 應該更新為：

> `dispatch()` no longer samples noisy RUL twice, but breakdown probability is still computed from noisy observed RUL rather than the underlying true RUL.

