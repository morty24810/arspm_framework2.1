# 結構化隨機 FJSP + Job-Conditioned Dust Feed 研究 TODO

## 1. 目標摘要

這份 TODO 的目的不是立刻改程式，而是把新的研究方向整理成可直接執行的待辦清單。

目標設定如下：

- machine 在單一 episode 內固定 `dust family`
- job 攜帶自己的 `dust_feed`
- machine 的退化速率由「machine intrinsic condition + job external load」共同決定
- FJSP 的 job-machine 對應不再是無條件隨機，而是改成 `family 相容約束下的結構化隨機`
- 第一階段先做 `A3 Medium + A4 Coarse`
- 規則層從一開始就必須 `A2-ready`
- 暫時完全不處理 `IM / CM` 可以清理多少

這條路線的正式結論是：

- 這不會破壞隨機性
- 真正被移除的是「不合理的無條件隨機」
- 要保留的是「相容集合內的結構化隨機」

## 2. 問題重述

### 2.1 舊假設

目前 repo 與 C. Ding 的簡化方式可以概括成：

- job 的 feasible machines 在 FJSP 中主要是無條件隨機抽樣
- machine 在退化層面固定綁定一條 replay curve
- C. Ding 不考慮 `job severity`
- job 不直接改變 machine 的退化速率

這樣做的好處是簡單，但問題是：

- 生產語義與退化語義是鬆耦合的
- machine 做了什麼 job，對退化的影響沒有被顯式建模
- machine 的材料/工況條件與加工類型沒有對應起來

### 2.2 新假設

新的研究問題改寫為：

- machine 先有固定的 `dust family`
- job 先屬於某個 family-compatible 的加工類別
- 每個 job 再帶一個 `dust_feed`
- job 被分派到 machine 後，退化速率依據該 machine 的 family 與該 job 的 feed 共同決定

也就是說：

- machine intrinsic condition：`dust family` 固定
- job-conditioned load：`dust_feed` 固定且由 job 攜帶
- 退化不是只看 machine ID，也不是只看固定 replay curve，而是看 machine 條件與 job 負載的交互

### 2.3 RUL 問題重述

第一階段的 RUL 問題不再沿用「某條 `Test` 序列的尾端 `RUL` 就是某台 machine 的固定命長」這種解讀，也不再把 machine 的退化理解成「固定 replay curve + 固定 lifespan」。

第一階段正式採用：

- `共同正規化壽命`

第一階段明確不採用：

- `共同絕對壽命常數`

原因是 dataset 顯示不同序列的總壽命差異很大，因此若直接規定所有 machine 在資料時間軸上共用同一個固定總壽命，會是一個過強簡化，不適合作為第一階段的正式假設。

第一階段的 RUL 定義應改寫為：

- 所有 machine 共享同一個 normalized health / failure budget
- machine 的起始健康度統一設為相同，例如 `h = 1`
- failure end-state 統一為 `h = 0`
- 真正需要學的是 `degradation rate function`
- `RUL` 應被視為 `latent health state` 的函數或 supervision，而不是直接綁定某條真實序列的尾端長度

這樣的好處是：

- 不再需要 `machine = one replay curve + one lifespan`
- job 可以透過 `dust_feed` 顯式影響退化速度
- machine-level 的 `dust family` 可以保留為固定材料/工況條件
- 後續若要把 `Hx / Hy` 改寫成 latent state 或 risk-based 邊界，會比較自然

## 3. 為什麼這不會破壞隨機性

### 3.1 三種隨機性要分開看

`無條件隨機`

- 任何 job 都可能分到任何 machine
- 這是目前簡化 FJSP 常見的做法
- 優點是方便
- 缺點是物理語義很弱

`結構化隨機`

- 先定義 `job family -> compatible machine pool`
- 再在 compatible pool 內隨機抽 feasible machines、隨機生成 route、隨機派工
- 這是第一階段推薦方案

`完全固定`

- 某類 job 永遠只去某幾台 machine
- 這不作為第一階段方案
- 原因是 agent 很容易記住 machine ID，而不是學到 family/load 結構

### 3.2 第一階段正式立場

第一階段採用：

- `結構化隨機`

這表示：

- 相容性規則是固定的
- 隨機性仍保留在相容集合內
- arrival、due date、processing time、派工順序都仍然可以是 stochastic

因此這不是把隨機性拿掉，而是把隨機性限制在物理上合理的範圍內。

## 4. 第一階段研究假設

- 研究對象先做 `A3 Medium + A4 Coarse`
- machine 在單一 episode 內固定 `dust family`
- job 在單一 job 內固定 `dust_feed`
- `dust_feed` 只使用 dataset 中真實觀測到的離散檔位
- `dust type` 第一階段不隨 job 改變
- 所有 machine 共享同一個 normalized failure budget
- machine 間差異主要體現在 job exposure history，而不是先體現在各自不同的總命長
- 第一階段不估計 machine-specific initial health variation，預設所有 machine 起點一致
- 第一階段只研究退化與排程的耦合，不處理 `IM / CM`
- `Flow_rate` 先作為退化模型的條件量，不作為 job family 的切換變數
- processing time、due date、arrival process 可以保留現有隨機生成邏輯，但要放在 family 相容約束之後

## 5. 資料支持證據

### 5.1 為什麼第一階段先做 A3 + A4

- `A3 Medium` 在 `Train/Test` 中都有 8 個可用 `dust_feed` 檔位
- `A4 Coarse` 在 `Train/Test` 中也都有 8 個可用 `dust_feed` 檔位
- `A2 Fine` 的 feed 覆蓋較少，因此不作第一階段主體，但必須保留未來擴充入口

第一階段可直接使用的 `dust_feed` 離散檔位如下：

- `A3`: `59.107236`, `79.246266`, `118.214472`, `158.492533`, `177.321707`, `236.428943`, `237.738799`, `316.985065`
- `A4`: `59.107236`, `79.246266`, `118.214472`, `158.492533`, `177.321707`, `236.428943`, `237.738799`, `316.985065`

### 5.2 第一階段不把 A2 當主體的原因

- `A2` 在 `Train` 中只有 4 個 feed 檔位
- `A2` 在 `Test` 中雖然有 5 個 feed 檔位，但仍少於 `A3/A4`
- 若第一階段直接同時處理 `A2/A3/A4`，研究空間會變大，但第一版不一定更清楚

### 5.3 目前已可直接使用的資料條件

- `Sampling = 10`
- `Δt = 0.1`
- `Dust_feed` 是離散外部負載條件
- `D10 / D50 / D90` 與完整 particle-size passing curve 已可直接取得
- 同一 dust type 下，`dust_feed` 變化確實會影響總壽命與壓差跨門檻時間

### 5.4 family 描述的最小粒徑摘要

- `A3 Medium`: `D10 = 2.606`, `D50 = 14.214`, `D90 = 59.827`
- `A4 Coarse`: `D10 = 4.467`, `D50 = 34.620`, `D90 = 104.761`
- `A2 Fine` 未來若加入，可沿用同一套 family 描述方式

### 5.5 為什麼不用共同絕對壽命

- `Test` 50 條序列的 `Time + RUL` 平均約 `115.8`，但最短約 `26.2`、最長約 `335.2`，變異很大
- A2 / A3 / A4 的平均總壽命層次明顯不同，代表 dust family 本身就和退化快慢高度相關
- 同一 dust type 下，不同 `dust_feed` 也會顯著拉開總壽命
- 因此資料支持的是「共同健康尺度 + 不同消耗速度」，不是「共同資料時間長度」

## 6. A2-ready 架構要求

第一階段雖然不啟用 `A2`，但 TODO 從一開始就必須是 `A2-ready`。

這表示：

- family 枚舉必須用一般形式書寫：`family = {A2, A3, A4}`
- machine pool 規則不能硬編碼成只支援 `A3/A4`
- job family 規則不能硬編碼成只支援 `A3/A4`
- 資料映射方式要寫成「每個 family 對應一組可用 feed 檔位」的一般形式
- 第一階段只是激活 `A3` 與 `A4`
- 第二階段可以在不推翻規則層的前提下啟用 `A2`

## 7. 研究 TODO

### A0. RUL 問題定義

- [ ] 明確廢除 `machine = one replay curve + one lifespan` 的 RUL 解讀
- [ ] 把第一階段 RUL 定義正式寫成 `共同正規化壽命尺度`
- [ ] 定義 machine-level fixed condition：`dust family`
- [ ] 定義 job-level external load：`dust_feed`
- [ ] 定義 `RUL` 為由退化狀態推導出的衍生量
- [ ] 寫清楚第一階段先不處理 machine-specific initial health dispersion

### A. 研究建模重述

- [ ] 明確定義 machine intrinsic condition：每台 machine 在 episode 內固定 `dust family`
- [ ] 明確定義 job-conditioned load：每個 job 固定一個 `dust_feed`
- [ ] 定義退化速率為 machine family 與 job feed 的交互結果
- [ ] 明確區分「machine 本質條件」與「job 外部負載」
- [ ] 明確說明第一階段不研究 `IM / CM`

### B. FJSP 相容性重寫

- [ ] 定義 `job family -> compatible machine pool`
- [ ] 明確禁止所有 job 無條件隨機抽所有 machine
- [ ] 將第一版相容性原則定義為 `結構化隨機`
- [ ] 保留 pool 內的隨機 feasible-machine 抽樣
- [ ] 保留 processing time、arrival、due date 的隨機性，但必須在 family 相容性之後生成
- [ ] 將 family-to-machine 規則寫成可配置映射，而不是硬編碼邏輯

### C. 資料映射策略

- [ ] 第一階段只用真實觀測到的離散 `dust_feed`
- [ ] `A3` job 只從 `A3` 的可用 feed 檔位選
- [ ] `A4` job 只從 `A4` 的可用 feed 檔位選
- [ ] 寫清楚未來若加入 `A2`，也遵循同一映射模式
- [ ] 不做連續 feed 插值
- [ ] 不做跨 family 的 dust type 切換

### D. 退化假設

- [ ] 不再把退化理解成 machine 固定綁一條單獨 replay curve 就足夠
- [ ] 把 job-level `dust_feed` 視為外部負載輸入
- [ ] 保留 machine-level `dust family` 作為材料/工況條件
- [ ] 把 `Differential_pressure` 視為可觀測退化訊號
- [ ] 把 `Differential_pressure` 的斜率也視為估計退化狀態的重要訊號
- [ ] 把 `Flow_rate` 視為條件量，而不是先驗忽略項
- [ ] 把 `Dust_feed`、`Flow_rate` 與壓差訊號視為估計 latent state 的輸入，而不是只作輔助欄位
- [ ] 把 latent degradation state 優先理解成 normalized health / clogging severity，而不是直接代表剩餘時間長短
- [ ] 把 latent degradation state 視為未來物理模型的核心，而不是只看 black-box RUL

### E. 驗證與比較

- [ ] 驗證同一 family 下，feed 升高是否通常對應更快退化
- [ ] 驗證在 `A3`、`A4` 兩類下這個方向是否一致
- [ ] 驗證結構化隨機相較於無條件隨機，是否更符合加工語義與物理語義
- [ ] 比較新假設相較於 C. Ding「job 不影響退化」的研究增益
- [ ] 確認 agent 學到的是 family/load 結構，而不是 machine ID 記憶

### F. 第二階段延伸

- [ ] 啟用 `A2`
- [ ] 允許部分 family overlap 或更複雜 machine specialization
- [ ] 視需要研究 `dust_feed` 插值
- [ ] 視需要讓 job 也影響 `Flow_rate` 或 processing burden
- [ ] 最後才回頭處理 `IM / CM`

## 8. 驗收清單

- [ ] 文檔明確回答「這不會破壞隨機性」
- [ ] 文檔明確採用 `結構化隨機`
- [ ] 文檔明確採用 `共同正規化壽命`，而非 `共同絕對壽命`
- [ ] 文檔明確寫出第一階段只做 `A3 + A4`
- [ ] 文檔明確寫出 `dust_feed` 只取真實觀測檔位
- [ ] 文檔明確寫出 machine 固定 family、job 固定 feed
- [ ] 文檔明確寫出 `RUL` 是衍生量，而不是 machine 固定命長標籤
- [ ] 文檔明確寫出第一階段學的是退化速率，而不是各 machine 的固定 lifespan
- [ ] 文檔明確寫出 `IM / CM` 不在本輪範圍
- [ ] 文檔明確指出現有 repo 的舊假設是無條件隨機 feasible machines + 固定 curve replay
- [ ] 文檔明確保留 `A2-ready` 條件

## 9. 預設假設

- 第一階段預設採用 `結構化隨機`
- 第一階段只激活 `A3` 與 `A4`
- 規則層從一開始就必須 `A2-ready`
- `dust_feed` 第一階段只用 dataset 中真實出現過的離散值
- 第一階段預設所有 machine 起始 normalized health 相同，差異由 job-conditioned degradation 累積形成
- 這份 TODO 只聚焦 `job-conditioned degradation`
- `IM / CM` 的物理恢復模型完全留到後面
