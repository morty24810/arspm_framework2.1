# 結構化隨機 FJSP + Operation-Conditioned Dust Feed 研究 TODO

## 1. 目標摘要

這份 TODO 的目的不是立刻改程式，而是把新的研究方向整理成可直接執行的待辦清單。

第一版正式設定如下：

- 總共 `6` 台 machine
- `A2 / A3 / A4` 三個 dust family 全部進入第一階段
- 每個 family 對應 `2` 台 machine
- machine 在單一 episode 內固定 `dust family`
- 同一個 job 的 route 可以跨 family
- 每個 `operation type` 固定一組 `dust family + dust_feed`
- 退化負載定義在 `operation-level`
- FJSP 的 job-machine 對應不再是無條件隨機，而是 `family 相容約束下的結構化隨機`
- 暫時完全不處理 `IM / CM` 可以清理多少

退化模型第一版正式採用：

- `共同正規化健康容量`
- `每個 family 一個 degradation model`
- `Train + Test` 全部軌跡用於 simulator calibration
- 同 family 的兩台 machine 第一版不加入 machine-specific 退化差異

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
- 每個 job 由多個 operations 組成
- 每個 `operation type` 先綁定一個固定的 `dust family`
- 每個 `operation type` 再綁定一個固定的 `dust_feed`
- operation 被分派到 machine 後，退化速率依據該 machine 的 family 與該 operation 的 feed 共同決定

也就是說：

- machine intrinsic condition：`dust family` 固定
- operation-conditioned load：`dust family + dust_feed` 固定且由 operation 攜帶
- 同一個 job 的不同 operations 可以落在不同 family 上
- 退化不是只看 machine ID，也不是只看固定 replay curve，而是看 machine 條件與 operation 負載的交互

第一版的 machine-family 配置固定為：

- machine `0, 1` 對應 `A2`
- machine `2, 3` 對應 `A3`
- machine `4, 5` 對應 `A4`

### 2.3 RUL / 退化問題重述

第一階段的 RUL 問題不再沿用「某條 `Test` 序列的尾端 `RUL` 就是某台 machine 的固定命長」這種解讀，也不再把 machine 的退化理解成「固定 replay curve + 固定 lifespan」。

第一階段正式採用：

- `共同正規化壽命`

第一階段明確不採用：

- `共同絕對壽命常數`

原因是 dataset 顯示不同序列的總壽命差異很大，因此若直接規定所有 machine 在資料時間軸上共用同一個固定總壽命，會是一個過強簡化，不適合作為第一階段的正式假設。

第一階段的退化與 RUL 定義應改寫為：

- 所有 machine 共享同一個 normalized health / failure budget
- machine 的起始健康度統一設為相同，例如 `h = 1`
- failure end-state 統一為 `h = 0`
- 真正需要學的是 `family-specific degradation rate function`
- `RUL` 應被視為 `latent health state` 的函數或 supervision，而不是直接綁定某條真實序列的尾端長度
- `A2 / A3 / A4` 各自擬合一個 degradation model
- 同 family 的兩台 machine 共用同一個退化模型，不額外加入 machine-specific health 偏移

這樣的好處是：

- 不再需要 `machine = one replay curve + one lifespan`
- operation 可以透過 `dust_feed` 顯式影響退化速度
- machine-level 的 `dust family` 可以保留為固定材料/工況條件
- 後續若要把 `Hx / Hy` 改寫成 latent state 或 risk-based 邊界，會比較自然

## 3. 為什麼這不會破壞隨機性

### 3.1 三種隨機性要分開看

`無條件隨機`

- 任何 operation 都可能分到任何 machine
- 這是目前簡化 FJSP 常見的做法
- 優點是方便
- 缺點是物理語義很弱

`結構化隨機`

- 先定義 `operation type -> dust family + dust_feed -> compatible machine pool`
- 再在 compatible pool 內隨機抽 feasible machines、隨機生成 route、隨機派工
- 這是第一階段正式方案

`完全固定`

- 某類 operation 永遠只去唯一一台 machine
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

- 研究對象直接採用 `A2 Fine + A3 Medium + A4 Coarse`
- machine 在單一 episode 內固定 `dust family`
- 同一個 job 的不同 operations 可以跨 family
- 每個 `operation type` 固定一組 `dust family`
- 每個 `operation type` 固定一組 `dust_feed`
- `dust_feed` 只使用 dataset 中真實觀測到的離散檔位
- 所有 machine 共享同一個 normalized failure budget
- machine 間差異主要體現在 operation exposure history，而不是先體現在各自不同的總命長
- 同 family 的兩台 machine 第一版不估計 machine-specific initial health variation，預設起點一致
- 第一階段只研究退化與排程的耦合，不處理 `IM / CM`
- `Flow_rate` 先作為退化模型的條件量，不作為 operation family 的切換變數
- processing time、due date、arrival process 可以保留現有隨機生成邏輯，但要放在 family 相容約束之後

## 5. Operation-Family / Machine-Pool 設計原則

- 每個 `operation type` 必須先綁定一個固定的 `dust family`
- 每個 family 固定對應 `2` 台 machine
- sibling machines 之間採 `部分重疊`
- 共享 operations 允許同時由同 family 的兩台 machine 加工
- 共享 operations 在兩台 machine 上的 `proc_time` 必須區隔，避免兩台 machine 完全等價
- 每台 machine 還要保留一部分不完全重疊的 operation coverage，形成同 family 內的專長差異
- 同一個 job 可以由多個 operation types 構成，因此 route 可以跨 family
- 第一版不要求 operation coverage 在三個 family 間完全對稱，但規則寫法必須對 `A2 / A3 / A4` 一般化

## 6. Processing Time 與退化語義對齊原則

- family 間只要求平均趨勢合理：更易堵塞的 family 平均加工更快
- 不把 `A2 < A3 < A4` 寫成每個 operation 都必須嚴格成立的硬規則
- 同 family 共享 operations 的差異優先由 processing time 反映
- `dust_feed` 與 `proc_time` 都是 operation-level 定義，但語義不同
- `dust_feed` 的作用是驅動退化負載
- `proc_time` 的作用是驅動排程負載與機器占用時間
- 第一版允許某些 operations 在 family 間有局部例外，只要整體平均趨勢仍合理

## 7. 資料支持證據

### 7.1 A2 / A3 / A4 都可作為第一階段 family

- `A3 Medium` 在 `Train/Test` 中都有 8 個可用 `dust_feed` 檔位
- `A4 Coarse` 在 `Train/Test` 中也都有 8 個可用 `dust_feed` 檔位
- `A2 Fine` 在 `Train` 中有 4 個可用 `dust_feed` 檔位，在 `Test` 中有 5 個可用 `dust_feed` 檔位
- 因此三個 family 都可以進第一階段，但 `A2` 的資料使用方式不能沿用原始 benchmark split 的狹義理解

### 7.2 A2 的 feed 覆蓋不對稱

- `A2` 在 `Train` 中可用 feed 為：`59.107236`, `79.246266`, `118.214472`, `158.492533`
- `A2` 在 `Test` 中可用 feed 為：`59.107236`, `177.321707`, `236.428943`, `237.738799`, `316.985065`
- `A2` 的 `Train` 與 `Test` feed 覆蓋幾乎不重合
- 因此若 `A2` 要進第一階段，family-specific degradation model 第一版必須把 `Train + Test` 都視為 simulator calibration data

### 7.3 目前已可直接使用的資料條件

- `Sampling = 10`
- `Δt = 0.1`
- `Dust_feed` 是離散外部負載條件
- `D10 / D50 / D90` 與完整 particle-size passing curve 已可直接取得
- 同一 dust type 下，`dust_feed` 變化確實會影響總壽命與壓差跨門檻時間

### 7.4 family 描述的最小粒徑摘要

- `A2 Fine`: `D10 = 1.774`, `D50 = 8.250`, `D90 = 35.534`
- `A3 Medium`: `D10 = 2.606`, `D50 = 14.214`, `D90 = 59.827`
- `A4 Coarse`: `D10 = 4.467`, `D50 = 34.620`, `D90 = 104.761`

### 7.5 為什麼不用共同絕對壽命

- `Test` 50 條序列的 `Time + RUL` 平均約 `115.8`，但最短約 `26.2`、最長約 `335.2`，變異很大
- A2 / A3 / A4 的平均總壽命層次明顯不同，代表 dust family 本身就和退化快慢高度相關
- 同一 dust type 下，不同 `dust_feed` 也會顯著拉開總壽命
- 因此資料支持的是「共同健康尺度 + 不同消耗速度」，不是「共同資料時間長度」

## 8. 一般化 Family 架構原則

- family 枚舉必須用一般形式書寫：`family = {A2, A3, A4}`
- machine pool 規則不能寫成三組彼此無關的特例
- operation family 規則不能硬編碼成只適用某一個 family
- 資料映射方式要寫成「每個 family 對應一組可用 feed 檔位」的一般形式
- 退化模型寫法要允許未來從 `每個 family 一個模型` 擴展到 `統一模型 + family 條件輸入`
- 第一版雖然固定採 `A2 / A3 / A4` 各兩台 machine，但規則層仍要能容納未來的部分 overlap、不同 machine count 與 manufacturing tolerance

## 9. 研究 TODO

### A0. RUL / 退化定義

- [ ] 明確廢除 `machine = one replay curve + one lifespan` 的 RUL 解讀
- [ ] 把第一階段 RUL 定義正式寫成 `共同正規化壽命尺度`
- [ ] 定義 machine-level fixed condition：`dust family`
- [ ] 定義 operation-level external load：`dust family + dust_feed`
- [ ] 定義 `RUL` 為由退化狀態推導出的衍生量
- [ ] 寫清楚第一階段採 `A2 / A3 / A4` 各一個 degradation model
- [ ] 寫清楚同 family twin machines 共用同一退化模型，不加 machine-specific health 偏移

### A1. Operation-Family 語義

- [ ] 定義每個 `operation type` 必須先綁定一個固定的 `dust family`
- [ ] 定義每個 `operation type` 必須再綁定一個固定的 `dust_feed`
- [ ] 寫清楚同一 job 的不同 operations 可以跨 family
- [ ] 明確區分「machine 本質條件」與「operation 外部負載」
- [ ] 明確說明第一階段不研究 `IM / CM`

### B. Machine Pool 與部分重疊規則

- [ ] 定義 `6 台 machine = A2×2 + A3×2 + A4×2`
- [ ] 定義 `operation type -> compatible machine pool`
- [ ] 明確禁止所有 operations 無條件隨機抽所有 machine
- [ ] 將第一版相容性原則定義為 `結構化隨機`
- [ ] 對每個 family 寫清楚 sibling machines 的 `部分重疊` 規則
- [ ] 明確規定共享 operations 在兩台 machine 上必須有不同 `proc_time`
- [ ] 保留 processing time、arrival、due date 的隨機性，但必須在 family 相容性之後生成

### C. Data-to-Operation 映射

- [ ] 第一階段只用真實觀測到的離散 `dust_feed`
- [ ] 寫清楚 `A2 / A3 / A4` 各自可用的 feed 檔位集合
- [ ] 把 `dust_feed` 的選取規則寫成 `operation type -> family-specific feed set`
- [ ] 寫清楚 `Train + Test` 在第一版中的角色是 simulator calibration data
- [ ] 特別標註 `A2` 的 `Train/Test` feed 覆蓋不對稱
- [ ] 不做連續 feed 插值

### D. Processing Time 趨勢與 Family 差異

- [ ] 定義 family 間 processing time 只要求平均上呈現合理趨勢
- [ ] 明確不把 `A2 < A3 < A4` 寫成所有 operations 的硬性單調排序
- [ ] 把同 family 內兩台 machine 的差異優先放在共享 operations 的 `proc_time` 上
- [ ] 把 `dust_feed` 與 `proc_time` 的角色分開描述
- [ ] 把 `Differential_pressure` 與其斜率視為估計退化狀態的重要訊號
- [ ] 把 `Flow_rate` 視為 family-specific degradation model 的條件量，而不是先驗忽略項
- [ ] 把 latent degradation state 優先理解成 normalized health / clogging severity，而不是直接代表剩餘時間長短

### E. 驗證與比較

- [ ] 驗證同一 family 下，feed 升高是否通常對應更快退化
- [ ] 驗證在 `A2`、`A3`、`A4` 三類下這個方向是否成立
- [ ] 驗證結構化隨機相較於無條件隨機，是否更符合加工語義與物理語義
- [ ] 驗證 family-specific degradation model 是否能支撐 `6 台 machine / 3 families` 的模擬設定
- [ ] 比較新假設相較於 C. Ding「job 不影響退化」的研究增益
- [ ] 確認 agent 學到的是 family/load 結構，而不是 machine ID 記憶

### F. 第二階段延伸

- [ ] 研究 `每個 family 一個模型` 是否要升級成 `統一模型 + family 條件輸入`
- [ ] 視需要加入 machine-specific manufacturing tolerance
- [ ] 視需要研究 `dust_feed` 插值
- [ ] 視需要擴展更複雜的 family overlap 與 machine specialization
- [ ] 最後才回頭處理 `IM / CM`

## 10. 驗收清單

- [ ] 文檔明確回答「這不會破壞隨機性」
- [ ] 文檔明確採用 `結構化隨機`
- [ ] 文檔明確採用 `共同正規化壽命`，而非 `共同絕對壽命`
- [ ] 文檔明確寫出 `6 台 machine = A2×2 + A3×2 + A4×2`
- [ ] 文檔明確寫出同一 job 的 operations 可跨 family
- [ ] 文檔明確寫出每個 operation type 固定 `dust family` 與 `dust_feed`
- [ ] 文檔明確寫出 sibling machines 採 `部分重疊`
- [ ] 文檔明確寫出共享 operations 需用 processing time 區隔
- [ ] 文檔明確寫出 `RUL` 是衍生量，而不是 machine 固定命長標籤
- [ ] 文檔明確寫出第一階段學的是 family-specific 退化速率，而不是各 machine 的固定 lifespan
- [ ] 文檔明確寫出 `Train + Test` 第一版都用於 calibration
- [ ] 文檔明確寫出 `A2` 的 feed 覆蓋不對稱
- [ ] 文檔明確寫出易堵塞 family 只要求平均上更快，而不是所有 operation 的硬性單調排序
- [ ] 文檔明確寫出 `IM / CM` 不在本輪範圍
- [ ] 文檔明確指出現有 repo 的舊假設是無條件隨機 feasible machines + 固定 curve replay

## 11. 預設假設

- 第一階段預設採用 `結構化隨機`
- 第一階段直接激活 `A2`、`A3`、`A4`
- 第一階段固定 `6` 台 machine，並按 `A2×2 + A3×2 + A4×2` 分配
- 第一階段 `dust_feed` 只用 dataset 中真實出現過的離散值
- 第一階段預設所有 machine 起始 normalized health 相同，差異由 operation-conditioned degradation 累積形成
- family-specific 退化模型的用途是 simulator / state model calibration，不是維持原始 benchmark split
- 這份 TODO 只聚焦 `operation-conditioned degradation`
- `IM / CM` 的物理恢復模型完全留到後面
