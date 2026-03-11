# C. Ding (2025) Maintenance Agent 設計整理

## 1. 方法定位

C. Ding 的 ARSPM 由兩個主要模組組成：

1. `PHM module`
   - 從監測資料中預測 machine 的 RUL
   - 把 RUL 與維修相關資訊抽象成 maintenance agent 的 state features
2. `MADDQN-based decision module`
   - `production agent` 決定派工規則
   - `maintenance agent` 決定維修動作

maintenance agent 的角色不是直接預測 RUL，而是接收 RUL 與維修歷史後，在即時決策點選擇 `DN / IM / CM`。

## 2. 與 maintenance 決策直接相關的問題假設

論文中與 maintenance agent 最直接相關的假設如下：

1. 維修時機器停止加工。
2. 維修只能在某道工序完成之後執行，不允許中斷加工去維修。
3. 機器 idle 時不會因其他因素繼續退化。
4. 不考慮不同 job 對機器退化嚴重度的差異。
5. 不考慮維修技能或維修資源限制。
6. 維修資源數量視為無限。
7. 每台機器只對應一種功能，不同機器在論文中有各自維修參數。

這些假設讓 maintenance agent 的決策集中在「何時修、修到什麼程度」，而不是「誰來修、能不能修、不同 job 造成多少磨損」。

## 3. PdM Region 設計

### 3.1 正規化 RUL

論文先把 RUL 正規化到 `[0, 1]`，代表剩餘壽命占總壽命的比例。maintenance agent 的門檻判斷都是在這個正規化空間進行。

### 3.2 三個區域

論文用兩個 RUL 門檻 `Hx` 與 `Hy` 定義三個區域：

- Region A: `H_t >= Hx`
  - 機器被視為正常，不啟動 maintenance scheduling。
- Region B: `Hy < H_t < Hx`
  - 進入 maintenance scheduling，agent 可在 `DN / IM / CM` 間做選擇。
- Region C: `H_t <= Hy`
  - 視為高度不可靠區域，必須立即執行 `CM`。

訓練時的主要門檻設定為：

- `Hx = 0.3`
- `Hy = 0.1`

論文也用其他門檻組合做 baseline maintenance scenarios，例如：

- MT1: `(Hx=0.3, Hy=0.1)`
- MT2: `(Hx=0.2, Hy=0.1)`
- MT3: `(Hx=0.25, Hy=0.1)`
- MT4: `(Hx=0.15, Hy=0.1)`

## 4. Action Space

maintenance agent 的 action space 為：

| 動作 | 含義 | 論文中的功能 |
| --- | --- | --- |
| `DN` | Do nothing | 不維修，讓機器繼續進入生產排程 |
| `IM` | Imperfect maintenance | 做不完全維修，把機器恢復到介於「as good as new」與「as bad as old」之間 |
| `CM` | Corrective maintenance | 直接更換/修復到完全恢復狀態 |

論文對三種動作的基本觀點：

- `DN` 沒有維修成本，但可能讓退化繼續累積。
- `IM` 比 `CM` 便宜、也更符合真實維修場景，但效果有限。
- `CM` 成本最高，通常在退化嚴重時使用。

## 5. State Space

論文 Table 3 為 maintenance agent 設計了 7 個 state features：

| 編號 | 狀態特徵 | 解釋 |
| --- | --- | --- |
| II-1 | The number of CM activities | 到目前為止的 CM 次數 |
| II-2 | The number of IM activities between two consecutive CM activities | 兩次 CM 之間累計的 IM 次數 |
| II-3 | The difference between the current time and the time of the last maintenance activity | 距離上一次維修活動的時間差 |
| II-4 | The estimated value of the current RUL | 當前估計 RUL |
| II-5 | The difference between the estimated value of the current RUL and RUL threshold `Hx` | 當前 RUL 與 `Hx` 的差值 |
| II-6 | The difference between the estimated value of the current RUL and RUL threshold `Hy` | 當前 RUL 與 `Hy` 的差值 |
| II-7 | The difference between the current time and the time when the RUL reaches `Hx` | 當前時間與 RUL 到達 `Hx` 時刻的差值 |

這 7 個特徵的核心思想很明確：

- 用 `II-4 ~ II-7` 表示目前退化位置與門檻相對關係
- 用 `II-1 ~ II-3` 表示維修歷史與維修節奏

對你後續改造最重要的一點是：這個 state space 幾乎完全建立在「RUL 已經被可靠預測」這個前提上，對原始感測量、材料、粉塵與負載機理幾乎沒有顯式建模。

## 6. Cost / Time / Reward 設計

### 6.1 維修時間設計

論文假設 IM 與 CM 的時間模型不同：

- `IM`
  - 維修時間會隨退化程度增加而變長
  - 論文把它建模成線性函數：基礎維修時間 `a_i` 加上一個隨退化增加的項 `b_i`
- `CM`
  - 維修時間是固定值 `MT_i^cm`

其核心思想是：

- 機器越晚修，IM 越耗時
- CM 雖然昂貴，但時間模型比較簡單，且恢復完整

### 6.2 維修成本設計

論文用每台機器的參數來描述維修成本：

- `CT_i^cm`: CM 的固定成本
- `w_i`: IM 的單位時間成本

因此 maintenance cost 可概括為：

- `CM cost` 由每台機器固定參數給定
- `IM cost` 由 `w_i × MT_i^im` 決定

Table 6 進一步給了每台機器的五類維修參數：

- `a_i`
- `b_i`
- `MT_i^cm`
- `CT_i^cm`
- `w_i`

### 6.3 Reward 機制

論文的總目標是同時最小化：

- `CT^T`：tardiness cost
- `CT^M`：maintenance cost

對 maintenance agent 而言，reward 設計的重點是「降低 maintenance cost」。

Algorithm 2 的邏輯可以摘要成：

- `DN`：
  - reward 設為 `0`
- `IM / CM`：
  - reward 依據「兩次 CM 之間的維修成本與正常運行時間間隔」來計算
  - 本質上是用 cost-per-operating-interval 的方式回饋 agent

就設計意圖而言，可以把它理解成：

- `DN` 不立即付出維修成本
- `IM` / `CM` 的 reward 帶有成本懲罰性質
- agent 要學的是：在不過早大修的前提下，把 IM / CM 插在足夠划算的位置

一個重要細節是：

- maintenance reward 不是在每一步立刻完全定義
- 它是在兩次 consecutive CM 之間累積，並在下一次 `CM` 發生後回填給這段區間內的維修動作

這讓 maintenance agent 的 credit assignment 帶有明顯延遲。

## 7. Online Decision Flow

Algorithm 3 可以整理成以下流程：

1. 新 job 持續進入系統。
2. 當某台 machine 在 decision point 變成 idle：
   - 先由 RUL prediction model 估計當前 `H_t`
3. 若 `H_t >= Hx`：
   - 啟動 production scheduling
   - maintenance agent 不介入
4. 若 `Hy < H_t < Hx`：
   - 啟動 maintenance scheduling
   - maintenance agent 觀察 state，從 `DN / IM / CM` 中選擇
5. 若在 Region B 選到：
   - `DN`：回到 production scheduling
   - `IM`：執行不完全維修，再進入下一狀態
   - `CM`：執行 corrective maintenance，並結算前一段維修區間 reward
6. 若 `H_t <= Hy`：
   - 不再交給 agent 自由選擇
   - 直接強制 `CM`

這個流程的核心結構是：

- production agent 決定「做哪個 job」
- maintenance agent 決定「這個時候要不要修」
- RUL 門檻負責在兩者之間切換決策主導權

## 8. 論文中的簡化假設與可修改點

如果你要在 C. Ding 的基礎上做更真實的模型，以下是最值得優先改的 simplification。

### 8.1 RUL 模型過度簡化

論文在 RUL prediction 部分做了幾個強簡化：

- 只用 `Differential_pressure` 當輸入特徵
- 只用 8 條選定序列中的一條 machine 5 做 train/validation
- 其他 7 條當 testing

這使 maintenance agent 接收到的 state，本質上建立在一個非常窄的 RUL estimator 上。

### 8.2 IM 的恢復規則是人工設定

論文因為每台 machine 只有一條 lifecycle，所以把 IM 直接建模成：

- 維修後壽命恢復到前一狀態剩餘壽命的 `80%`
- 若連續多次 IM，就反覆套用這個比例

這不是原始 dataset 的屬性，而是作者為了模擬 maintenance effect 所加的規則。

### 8.3 加噪 augmentation 不是原始資料

論文為了模擬 real-time decision-making，對資料加噪做 augmentation。這同樣不是 dataset 自帶訊息，而是作者為了把靜態 life-cycle data 變成可在線決策的場景所做的額外假設。

### 8.4 退化來源沒有顯式機理

論文的 maintenance agent state 幾乎都建立在 RUL 與門檻差值上，而不是建立在：

- dust type
- dust feed
- flow rate
- cumulative loading
- filter material properties

如果你想做更物理的模型，這裡是最值得擴充的地方。

### 8.5 不考慮 job severity

論文假設 job 不會造成不同程度的退化，這讓生產與退化之間的耦合被大幅簡化。若你後續要把物理退化與排程更緊密結合，這一點可能需要被放寬。

## 9. 對你後續修改最有用的摘要

如果把 maintenance agent 當作可被改造的模組，C. Ding 的原始設計可以濃縮成：

- `輸入`：RUL 與維修歷史的低維摘要
- `決策空間`：`DN / IM / CM`
- `切換邏輯`：靠 `Hx / Hy` 將機器分區
- `維修效果`：IM 線性變差、CM 完全恢復
- `學習目標`：降低 maintenance cost，同時避免在低 RUL 區域拖到必須 CM

而你最值得優先改的地方則是：

1. 讓 state 不只依賴 black-box RUL，而是顯式帶入粉塵、流量、loading 與材料資訊。
2. 重新定義 IM/CM 對健康狀態的影響，不要只用固定 `80% reset`。
3. 重新設計 reward，讓它不只反映維修成本，也能反映堵塞風險、壓差成長與物理負載累積。
