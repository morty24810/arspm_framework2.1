# 用資料驅動的物理隱狀態替代 `Hx/Hy` 與 `IM=80%` 的可行性研究

## 1. 研究目的

這份 memo 的目的不是提升 RUL 預測準確率，而是評估是否可以用一個「資料驅動、但具有物理解釋的隱狀態模型」，替代目前系統與 C. Ding (2025) 中的兩類人為控制先驗：

- 固定門檻 `Hx / Hy`
- `IM` 後健康狀態固定恢復為上一基線的 `80%`

研究核心問題是：

1. 現有 dataset 能否支持建立「未維修退化」的物理或半物理隱狀態？
2. 這個隱狀態能否進一步替代 `Hx/Hy` 與 `IM=80%`？
3. 哪些部分可以從資料學到，哪些部分一定要靠模型假設，哪些部分需要額外資料才可識別？

先給結論：

- `資料直接支持`：這個方向是合理的，因為它可以把 maintenance decision 從「硬門檻 + 固定 reset」改寫成「狀態估計 + 狀態轉移 + 風險/成本決策」。
- `資料直接支持`：現有 dataset 足以支撐「未維修退化動力學」與「dust/feed/flow 對堵塞速度的影響」研究。
- `需要額外資料才可識別`：單次 `IM` 真正清除了多少粉塵、恢復了多少通透性，不能由現有 dataset 直接唯一決定，因為資料幾乎沒有真實維修介入前後的觀測。
- `需要模型假設`：若要先替代 `IM=80%`，必須把 `IM` 寫成某種 latent-state reduction operator，並用半物理假設或外部校準去限制其參數範圍。

## 2. 目前系統審計：哪些地方仍是控制先驗

### 2.1 Repo 目前對 `Hx/Hy` 的用法

目前專案內的 maintenance 決策不是由物理狀態自然推導，而是仍然明確依賴 `Hx / Hy`：

- [config.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/config.py) 定義 `Hx=0.3`、`Hy=0.1` 與 `ENFORCE_REGION_POLICY=True`
- [run_experiment.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/run_experiment.py) 中：
  - `validate_region_thresholds`
  - `allowed_actions_by_region`
  - `enforce_action_by_region`
  - `build_policy_label`
  都直接把 `Hx/Hy` 當作控制規則
- [src/env.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/src/env.py) 中：
  - `maintenance_decision_point` 會記錄第一次穿越 `Hx` 的時間
  - `_enforce_region_action` 在 `h > Hx` 時只允許 `DN`，在 `h < Hy` 時強制 `CM`
  - `failure_prob` 也直接以 `Hy` 為 sigmoid 中心構造失效機率

結論：

- `資料直接支持`：目前 repo 的 maintenance routing 仍然明確以 `Hx/Hy` 作為顯式決策邊界。
- `需要模型假設`：若要移除 `Hx/Hy`，就必須提供另一個能接管「何時進入 DN / IM / CM 決策區」的量，例如 posterior risk、expected cost-to-go 或 latent clogging state。

### 2.2 Repo 目前對 `IM=80%` 的用法

目前專案中 `IM_RESET=0.80` 雖然仍出現在 config，但註解已標示為 legacy；真正起作用的是 [src/env.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/src/env.py) 裡的硬編碼幾何更新：

- `apply_maintenance`：
  - `target_rul = 0.8 * m.maint_rul_baseline`
- `generative_step`：
  - `baseline_rul = 0.8 * baseline_rul`

也就是說，目前活躍邏輯不是「讀 config 中某個可校準參數」，而是直接假設：

- 每做一次 `IM`，健康基線乘上 `0.8`
- `CM` 直接重設到 `1.0`

結論：

- `資料直接支持`：目前 repo 的 `IM` 效果本質上是固定幾何比例，並非由資料辨識出的維修恢復模型。
- `需要模型假設`：若要替換 `0.8`，應該把 `IM` 改寫成 latent state 的轉移算子，而不是單純換另一個固定數字。

### 2.3 C. Ding 的原始設計與 repo 的差異

根據 [c_ding_maintenance_agent.md](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/c_ding_maintenance_agent.md)，C. Ding 的 maintenance agent 也是建在類似邏輯上：

- `Hx / Hy` 決定 Region A/B/C
- `DN / IM / CM` 是離散動作
- `IM` 恢復規則靠人工設定
- state 幾乎完全建立在 `RUL` 與維修歷史上

repo 雖然已經比 C. Ding 多了額外特徵，例如：

- `dh`
- `eta`
- `slack_pressure`
- `local_urgency`
- `risk_t`
- `pf_dn / pf_im / pf_cm`

但 maintenance transition 這一層仍然不是物理解釋模型，而是控制式先驗。

結論：

- `資料直接支持`：目前 repo 雖然比 C. Ding 多了一些 decision features，但核心 transition logic 仍然是人工設計。
- `需要模型假設`：若你的目標是替換控制先驗，真正該被重寫的是「健康狀態如何演化」與「維修如何作用在健康狀態上」，而不只是再加更多 heuristic features。

## 3. 可觀測量 vs 不可觀測量

### 3.1 Dataset 直接提供的可觀測量

- `Differential_pressure`
- `Flow_rate`
- `Time`
- `Dust_feed`
- `Dust`
- `RUL`（只在 `Test` 中提供）

這些量能支持的事情：

- `資料直接支持`：研究未維修退化曲線
- `資料直接支持`：比較不同 dust/feed/flow 條件下的堵塞快慢
- `資料直接支持`：構造某種 latent degradation state 的候選形式，並檢查它能否解釋壓差與壽命

### 3.2 Dataset 未直接提供的關鍵量

現有資料沒有直接量到以下變數：

- 真實沉積粉塵量
- 濾餅厚度或濾材孔隙率變化
- 真實 permeability / resistance 內部狀態
- 真實維修前後殘留粉塵量
- 真實 IM 強度與維修效果
- 維修後壓差恢復曲線
- 明確可解碼的 `Sampling` Hz 數值

這些量的地位要分清楚：

- `需要模型假設`：沉積粉塵量、cake resistance、permeability loss 可以作為 latent state 候選，但在現有資料裡不是直接量測值。
- `需要額外資料才可識別`：IM 後殘留粉塵量與真實維修效果，沒有維修介入觀測就無法唯一識別。
- `需要額外資料才可識別`：若你要把「一次 IM 清掉多少粉塵」當作可驗證物理量，必須有維修前後配對觀測或額外實驗。

## 4. 可識別性分析：資料能學到什麼、學不到什麼

### 4.1 可由現有資料直接支持的部分

- `資料直接支持`：未維修退化動力學
  - dataset 主要就是未維修壽命曲線
  - 這很適合拿來識別「在 dust/feed/flow 條件下，堵塞狀態如何隨時間增加」
- `資料直接支持`：dust / feed / flow 對堵塞速度的影響方向
  - 你在 `dataset_analysis.md` 已經看到 A2/A3/A4 與不同 feed 下的壽命差異
- `資料直接支持`：latent clogging state 的候選形式
  - 例如某個單調上升的 clog-load、resistance 或 permeability-loss 狀態
  - 只要它能透過 observation model 連到 `Differential_pressure` 與 failure/RUL，就有研究價值

### 4.2 只能間接校準、不能直接唯一決定的部分

- `需要模型假設`：單次 IM 的實際清除比例
  - 因為 dataset 幾乎沒有維修介入後的真實曲線
  - 你可以假設 `x^+ = (1-rho)x` 或 `x^+ = x - delta(u)`，但 `rho` / `delta` 不是資料直接給的
- `需要模型假設`：不同 maint 強度對健康恢復的真實函數
  - 例如 IM 是線性恢復、飽和恢復、還是與當前堵塞量非線性相關，現有資料無法直接分辨
- `需要模型假設`：latent state 與 `Differential_pressure` 的觀測方程形式
  - 例如 `DP = f(flow, x, dust_type)` 要採線性、對數、阻力疊加還是別的形式，都需要建模選擇

### 4.3 需要額外資料才可識別的部分

- `需要額外資料才可識別`：一次 IM 真實清除掉多少粉塵
- `需要額外資料才可識別`：維修後立即與短期內的壓差恢復曲線
- `需要額外資料才可識別`：維修強度與維修工時/材料消耗的真實對應
- `需要額外資料才可識別`：不同維修策略是否造成不同的後續退化速率

這一段其實就是本研究最重要的邊界：

- `資料直接支持`：你可以用 dataset 把「未維修的退化機理」建得更物理、更可解釋。
- `需要模型假設`：你可以把 `IM` 寫成物理狀態轉移算子。
- `需要額外資料才可識別`：但你不能宣稱「這個算子就是現實中唯一正確的 IM 清粉塵比例」，除非未來有維修介入資料去驗證。

## 5. 候選建模路線比較

### 5.1 路線 A：隱狀態 `clog-load / permeability-loss` 模型

這是推薦路線。

核心想法：

- 定義一個隱狀態 `x_t`，代表堵塞程度、沉積負荷或通透性損失
- 用狀態動力學描述未維修退化：
  - `x_{t+1} = x_t + g(dust, dust_feed, flow, t, ...)`
- 用觀測方程連接到壓差：
  - `DP_t = h(flow_t, x_t, dust_type, ...)`
- 用失效條件或 hazard 連接到 RUL：
  - `RUL_t = psi(x_t, flow_t, dust_type, ...)`

如果將來有維修：

- `CM` 可視為 `x^+ = 0` 或接近乾淨狀態
- `IM` 可視為 `x^+ = T_IM(x, u)`，其中 `u` 是維修強度

優點：

- `資料直接支持`：非常符合 dataset 的原始物理語境
- `資料直接支持`：同時能處理 `Hx/Hy` 與 `IM` 兩個問題
- `需要模型假設`：需要自己設計 latent state 與 observation model

結論：

- `資料直接支持`：這是最適合你目前目標的主路線。

### 5.2 路線 B：風險 / 危害率模型

核心想法：

- 不直接建 clog-load，而是學一個 failure risk / hazard / cost-to-go
- 用這個量取代 `Hx/Hy`，讓 decision boundary 由風險與成本決定

它能解決的問題：

- `資料直接支持`：可替代固定 `Hx/Hy`

它不能完整解決的問題：

- `需要模型假設`：對 `IM` 的真實物理效果仍然沒有直接答案
- `需要模型假設`：更像 decision layer replacement，而不是 maintenance physics replacement

結論：

- `資料直接支持`：如果你的短期目標只是不想再用硬閾值，這條路很實用。
- `基於研究目標`：但它無法完整回答「一次 IM 清除多少堵塞狀態」這個問題。

### 5.3 路線 C：維修轉移模型

核心想法：

- 不急著完整建 observation physics
- 先把維修建成顯式 state transition operator

例如：

- `x^+ = (1-rho_u) x`
- `x^+ = x - delta_u`
- `x^+ = x * exp(-k_u)`

其中 `u` 代表維修強度、工時或維修等級。

這條路的優點：

- `需要模型假設`：可以直接替換 `IM=80%`
- `需要模型假設`：便於後續和 RL action space 接軌

限制：

- `需要額外資料才可識別`：如果沒有維修介入資料，`rho_u` 或 `delta_u` 只能靠先驗範圍或外部校準限制

結論：

- `需要模型假設`：這條路適合做中間層，把 maintenance 從固定比例改成可校準轉移，但仍需與路線 A 或外部資料結合。

## 6. 對 maint agent 的含義

如果採用 latent-state 路線，未來 maint agent 的 state 不應再只圍繞 `RUL/Hx/Hy`，而應該改成更接近物理或半物理意義的量。

候選 state 類型：

- `資料直接支持`：當前 `Differential_pressure`
- `資料直接支持`：壓差成長速率、局部斜率、短期趨勢
- `資料直接支持`：`Flow_rate`
- `資料直接支持`：`Dust_feed`
- `資料直接支持`：`Dust` 類型
- `需要模型假設`：當前 posterior clog-load / cake resistance / permeability-loss
- `需要模型假設`：若執行 `IM`，預期可回復多少 latent state
- `需要模型假設`：短期 failure risk 或 expected remaining deposit capacity
- `需要模型假設`：`DN / IM / CM` 對未來 downtime 與 cost 的預測值

如果做到這一步，maint agent 的決策會更像：

- 在某個估計出的堵塞狀態 `x_t` 下
- 比較 `DN / IM / CM` 三種 action 的預期狀態轉移、失效風險、downtime 與成本

而不是：

- 先看 `h > Hx` 還是 `h < Hy`
- 再依區域硬切 action 空間

結論：

- `資料直接支持`：更多輸入層資訊確實能讓 maint RL state 更清晰。
- `需要模型假設`：真正關鍵不是「加多少 feature」，而是這些 feature 是否能被組織成一個可解釋的狀態轉移模型。

## 7. 最小可行研究路線

### 第 1 階段：先用無維修曲線擬合 latent clogging dynamics

目標：

- 用現有 `Train/Test` 曲線建立未維修退化模型

建議產出：

- 一個單調上升的 latent clogging state 候選
- 一個把 `Dust / Dust_feed / Flow_rate` 連到退化速度的動力學方程
- 一個把 latent state 連到 `Differential_pressure` 與 failure/RUL 的觀測/終止方程

結論：

- `資料直接支持`：這一步是現有 dataset 最能支持、也最值得先做的部分。

### 第 2 階段：把 `IM` 定義為 latent state reduction operator

目標：

- 不再用固定 `0.8 reset`
- 改成 `x^+ = T_IM(x, u)`

建議做法：

- 先不追求唯一真值
- 先根據工程合理性設定一族 operator
- 再用 simulation consistency、maintenance cost 與後續決策表現去篩選合理範圍

結論：

- `需要模型假設`：這一步可以先做，但不能宣稱已由 dataset 唯一驗證。

### 第 3 階段：把 `Hx/Hy` 改寫成軟邊界

目標：

- 不再使用固定 `Hx/Hy`
- 改由以下任一種量導出 decision boundary：
  - posterior risk
  - expected cost-to-go
  - latent state safety margin
  - predicted post-maintenance gain

結論：

- `需要模型假設`：這一步是控制層改寫
- `資料直接支持`：它可以建立在第 1 階段得到的 latent degradation model 上

## 8. 限制與需求

若你真正要把「一次 IM 清除多少粉塵」做成可驗證量，而不只是模型中的可調參數，至少需要以下額外資料或實驗之一：

- `需要額外資料才可識別`：真實維修前後的配對量測
- `需要額外資料才可識別`：維修強度或維修等級標記
- `需要額外資料才可識別`：維修後短時間內的壓差恢復曲線
- `需要額外資料才可識別`：濾芯清潔/更換後的實際殘留狀態或粉塵量測
- `需要額外資料才可識別`：多次維修後的後續退化曲線

如果沒有這些資料，合理的研究定位應該是：

- `資料直接支持`：建立未維修退化的資料驅動物理隱狀態
- `需要模型假設`：把 `IM/CM` 表示成可解釋、可校準的狀態轉移
- `需要額外資料才可識別`：未來再用維修介入資料驗證維修 operator 的真實性

## 9. 最後結論

- `資料直接支持`：你想做的方向是合理的，而且比單純追求 RUL 預測精度更對題。
- `資料直接支持`：這個 dataset 足夠支撐「未維修退化機理」的資料驅動物理建模。
- `需要模型假設`：要替代 `IM=80%`，可以先把 `IM` 改寫成 latent-state reduction operator。
- `需要模型假設`：要替代 `Hx/Hy`，可以把 decision boundary 改成由 risk / cost / posterior state 派生的軟邊界。
- `需要額外資料才可識別`：但「一次 IM 真正清掉多少粉塵」不能由現有 dataset 直接唯一識別。

因此，最合理的研究定位不是：

- 「用 dataset 直接反推出真實 IM 清潔比例」

而是：

- 「先用 dataset 建立可解釋的未維修退化隱狀態，再把 maintenance 建成可校準的狀態轉移，最後用未來維修介入資料去驗證與修正」
