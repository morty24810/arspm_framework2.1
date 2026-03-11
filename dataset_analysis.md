# Preventive to Predictive Maintenance Dataset 分析與論文對齊驗證

## 1. 分析目的

本報告同時對齊三類證據：

1. 原始資料集說明文件 `Preventive to Predictive Maintenance dataset.pdf`
2. 專案根目錄中的 `Train_Data_CSV.csv` 與 `Test_Data_CSV.csv`
3. 論文 `C. Ding_Adaptive real-time scheduling for production and maintenance: Integrating RUL prediction with multi-agent deep reinforcement learning (2025)`

目標不是重現論文模型，而是回答三個問題：

- 這個 dataset 在原始設計上到底要解什麼任務？
- 本地 CSV 與 PDF 描述是否一致？
- 2025 論文如何重用這個 dataset，哪些地方可驗證，哪些地方其實是作者自行加上的實驗設定？

## 2. 原始資料集與本地 CSV 概況

### 2.1 原始資料集背景

依照原始 PDF，這個資料集來自濾材堵塞的退化過程，用來模擬固體顆粒與氣體分離時，濾材在壓差上升下的壽命變化。PDF 明確給出以下背景：

- 觀測對象是濾材的 clogging / degradation 過程。
- 測量變數至少包含 `Differential_pressure`、`Flow_rate`、`Time`。
- 設定參數包含 `Dust`、`Dust_feed`，其中 `Dust_feed` 表示單位時間的加塵量。
- 失效條件是濾材兩端壓差超過 `600 Pa`。
- 資料集包含 `Train` 與 `Test` 兩部分，各 `50` 條生命試驗。

PDF 對 `Test` 有兩種表述：

- 一處摘要式描述說 `Test` 是 complete run-to-failure measurements。
- 任務段與資料結構段則更清楚地指出：`Test` 提供帶有 RUL ground truth 的 right-censored prediction task。

從本地 CSV 觀察，後者更符合實際資料形態，因為 `Test_Data_CSV.csv` 的各序列尾端 `RUL` 仍大於 0，並未直接觀測到 `RUL = 0` 的真正失效點。

### 2.2 本地 CSV 欄位與任務結構

| 檔案 | 欄位 |
| --- | --- |
| `Train_Data_CSV.csv` | `Data_No`, `Differential_pressure`, `Flow_rate`, `Time`, `Dust_feed`, `Dust` |
| `Test_Data_CSV.csv` | `Data_No`, `Differential_pressure`, `Flow_rate`, `Time`, `Dust_feed`, `Dust`, `RUL` |

關鍵觀察：

- `Train` 沒有 `RUL` 標籤。
- `Test` 多出 `RUL` 欄位，對應預測任務的 ground truth。
- PDF 資料結構中曾列出 `Sampling`，但目前兩份 CSV 都沒有這一欄，表示這是由原始結構轉成扁平 CSV 時省略掉的欄位。

### 2.3 本地 CSV 統計結果

| 檔案 | 總列數 | 序列數 (`Data_No`) | 缺值 | 序列長度 min / median / mean / max | 序列終止時間 min / median / mean / max |
| --- | ---: | ---: | ---: | --- | --- |
| `Train_Data_CSV.csv` | 39,420 | 50 | 0 | 335 / 669.0 / 788.4 / 1794 | 33.5 / 66.9 / 80.79 / 179.4 |
| `Test_Data_CSV.csv` | 39,414 | 50 | 0 | 150 / 558.5 / 788.28 / 2581 | 15.0 / 55.85 / 78.85 / 258.1 |

補充特徵：

- `Train` 的粉塵類型分布（以序列數計）：
  - A3 Medium: 26
  - A2 Fine: 11
  - A4 Coarse: 13
- `Test` 的粉塵類型分布（以序列數計）：
  - A3 Medium: 24
  - A2 Fine: 14
  - A4 Coarse: 12
- `Dust_feed` 在本地 CSV 中不是連續變量，而是 8 個離散設定值。
- 單一 `Data_No` 內的 `Dust` 與 `Dust_feed` 都保持固定，不會在同一條生命序列中跳變。

### 2.4 對 `Train` / `Test` 任務語義的直接驗證

本地資料對原始任務語義的支持度很高：

- `Train`：
  - 每條序列是單一生命試驗的截斷觀測。
  - 大部分序列末端壓差並未接近 600 Pa。
  - 只有 `5/50` 條序列的最大壓差達到或超過 `600 Pa`，`6/50` 條達到或超過 `580 Pa`。
  - 這與 PDF 所說「定期更換前的觀測」一致，符合 right-censored training data 的解讀。
- `Test`：
  - `RUL` 對所有 50 條序列都單調遞減，且每一步固定減少 `0.1`。
  - 對所有 50 條序列，`Time + RUL` 都維持常數。
  - 這代表 `Test` 更像是「在某個截尾時間點開始提供觀測，並附上剩餘壽命真值」的 prediction task，而不是單純從時間 0 跑到失效點的完整原始歷程。

## 3. 與 Ding et al. (2025) 的對齊結果

### 3.1 核心結論

1. 論文確實引用了 Hagmeyer 等人的公開濾材堵塞 RUL dataset，且正確指出原始可觀測變數包含壓差與流量。
2. 論文沒有直接沿用原始 dataset 的 `Train 50 + Test 50` 挑戰設定；它是把其中 8 條序列重解釋成 8 台機器，再自行定義 train / validation / test。
3. 根據壽命數值逐筆比對，論文的 8 台「機器壽命」不是原始完整壽命，而是與 `Test_Data_CSV.csv` 各對應序列的**首筆 `RUL` 值**一致（乘上 10 後得到論文中的整數壽命）。
4. 因此，論文中的 RUL 子任務比較像是「從原始 `Test` 中挑 8 條帶標籤序列，重建一個新的 8-machine benchmark」，而不是直接遵守原始資料集的官方 train/test protocol。

### 3.2 對齊 / 驗證矩陣

| 論文說法 | 原始資料集 PDF 是否支持 | 本地 CSV 是否支持 | 結論標籤 | 說明 |
| --- | --- | --- | --- | --- |
| 資料來自濾材堵塞的實務退化過程，觀測變數包含 `flow rate` 與 `differential pressure` | 是 | 是 | `已驗證` | 原始 PDF 與本地 CSV 均支持這一點。 |
| dataset 的目標是準確預測 RUL，以支援 predictive maintenance | 是 | 是 | `已驗證` | PDF 任務段明確說明此目標；本地 `Test` 也確實提供 `RUL`。 |
| 原始資料可被視為 8 台機器，其中使用 `Data_No = 4, 8, 11, 17, 18, 23, 28, 49` 代表 machine 1~8 | 否 | 部分支持 | `部分驗證` | PDF 沒有「8 台機器」的原始結構；本地 CSV 只證明這 8 個 `Data_No` 存在。將它們重映射成 8 台機器是作者的實驗抽樣。 |
| 8 台機器的 lifespans 為 `626, 1217, 1267, 1993, 2631, 1285, 2936, 2819` | 否 | 部分支持 | `表述不精確` | 這些數值與本地 `Test` 中對應序列的首筆 `RUL × 10` 對齊，但不等於完整壽命，也不等於 `Time + RUL` 推回的總生命長度。 |
| 這 8 條序列包含 complete operating lifecycle information | 部分支持 | 否 | `表述不精確` | PDF 自身對 `Test` 有「run-to-failure」與「right-censored + RUL」兩種表述；本地 CSV 顯示尾端 `RUL` 仍大於 0，更符合後者。 |
| `Differential_pressure` 被選為 RUL 模型的唯一輸入特徵 | 否 | 是 | `已驗證` | 本地 CSV 確實包含壓差欄位，因此此設定可行；但這是作者的特徵選擇，不是 dataset 的原始限制，因為 `Flow_rate`、`Dust_feed`、`Dust` 也都可用。 |
| machine 5 的資料拿來做 train / validation，其他 7 台做 testing | 否 | 否 | `無法直接驗證` | 這是論文自行定義的切分策略，原始 PDF 與本地 CSV 都不包含這個 split。 |
| IM 會把剩餘壽命恢復到前一次值的 80%，CM 則完全恢復 | 否 | 否 | `無法直接驗證` | 這是維修模擬規則，不是資料集原生欄位或標註。 |
| 為了模擬即時決策，作者對資料加噪做 augmentation | 否 | 否 | `無法直接驗證` | 原始資料中沒有 augmentation 訊號，這是作者後處理。 |
| 原始資料結構包含 `Sampling` | 是 | 否 | `已驗證` | PDF 結構描述包含 `Sampling`，但本地 CSV 扁平導出後沒有保留。 |
| 原始 `Train` 是定期更換前的觀測，`Test` 是帶 RUL 的 prediction task | 是 | 是 | `已驗證` | 本地 `Train` 無 `RUL`、`Test` 有 `RUL`，且 `Test` 滿足 `Time + RUL = constant`。 |

### 3.3 論文所選 8 條序列的具體對應

下表把論文中的 machine 1~8，對齊到本地 `Test_Data_CSV.csv` 的序列。論文給出的 lifespan 與本地資料最接近的對應方式，是「首筆 `RUL × 10`」。

| 論文 machine | 對應 `Data_No` | 論文 lifespan | `Test` 首筆 `Time` | `Test` 首筆 `RUL` | `Time + RUL` 常數 | 對齊判讀 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M1 | 4 | 626 | 0.2 | 62.6 | 62.8 | 論文值對齊首筆 `RUL × 10`，不是完整總壽命 |
| M2 | 8 | 1217 | 0.1 | 121.7 | 121.8 | 同上 |
| M3 | 11 | 1267 | 0.1 | 126.7 | 126.8 | 同上 |
| M4 | 17 | 1993 | 0.1 | 199.3 | 199.4 | 同上 |
| M5 | 18 | 2631 | 0.1 | 263.1 | 263.2 | 同上 |
| M6 | 23 | 1285 | 0.1 | 128.5 | 128.6 | 同上 |
| M7 | 28 | 2936 | 0.1 | 293.6 | 293.7 | 同上 |
| M8 | 49 | 2819 | 0.1 | 281.9 | 282.0 | 同上 |

這張表帶出兩個重要推論：

- 論文的 lifespan 值是以 `0.1` 為時間粒度，直接取 `Test` 首筆 `RUL` 後再乘以 10。
- 這些值**不是**原始完整壽命；若改用 `Time + RUL` 還原，對應整數會變成 `628, 1218, 1268, 1994, 2632, 1286, 2937, 2820`。

另外，同一組 `Data_No` 在 `Train_Data_CSV.csv` 中的序列長度分別是 `897, 1794, 623, 669, 669, 897, 598, 446`，與論文的 `626, 1217, 1267, 1993, 2631, 1285, 2936, 2819` 完全不對應。這進一步支持：論文的 8-machine 壽命定義並不是來自原始 `Train`，而是更接近 `Test` 序列起點的剩餘壽命標籤。

換句話說，論文對「machine lifespan」的定義，比較接近「該截尾序列起點的剩餘壽命」，不是原始生命週期全長。

### 3.4 對論文資料使用方式的合理推斷

根據上面的逐筆對齊，可做出一個相當穩固的推斷：

- 論文的 8-machine RUL 子任務，主要是從原始 `Test_Data_CSV.csv` 抽取 8 條**帶標籤**序列。
- 論文後續又把其中一條（machine 5，對應 `Data_No=18`）拿來做 `80% train + 20% validation`，其餘 7 條當作 testing。
- 這個流程與原始 dataset 的官方使用方式不同，因為原始設計是 `Train` 供訓練、`Test` 供預測與驗證；論文則是重新切出自己的 benchmark。

這個差異本身不代表論文做法一定錯，但若後續引用這篇論文，應避免把它描述成「直接遵循原始 dataset 官方 train/test protocol」。

## 4. 資料品質與復用注意事項

### 4.1 本地 CSV 的資料品質提醒

- `Train_Data_CSV.csv` 的 `Data_No=28` 有明顯時間排序異常：
  - 首筆時間是 `157.2`
  - 之後直接跳回 `0.2, 0.3, 0.4 ...`
  - 該序列前 5 筆時間為 `157.2, 0.2, 0.3, 0.4, 0.5`
- 因此：
  - `Train` 中只有 `49/50` 條序列保持嚴格時間遞增與固定 `0.1` 步長
  - `Test` 則是 `50/50` 條都符合這個特性

若任何模型直接依賴序列順序，應先對 `Train` 做排序或清洗，至少要單獨處理 `Data_No=28`。

### 4.2 原始任務中的 censoring 不能被忽略

- 原始 PDF 的關鍵挑戰是：如何利用 right-censored training data 來做好 RUL prediction。
- 本地 `Train` 的末端壓差大多不到 600 Pa，正好支持這個設計。
- 若把這些序列重新解讀成完整 run-to-failure，不僅會弱化資料集原始意圖，也可能高估模型所看到的失效資訊量。

### 4.3 「8 台機器」是任務層抽象，不是原始資料語義

在原始資料中，基本單位是生命試驗序列 `Data_No`，不是工廠中的固定 machine ID。論文把 8 條序列映射為 8 台機器，這是為了後續排程與維修策略模擬而建立的任務層抽象。引用時應明確寫成：

- 「論文自原始資料中挑選 8 條序列，重建 8 台機器的實驗場景」

而不是：

- 「原始 dataset 本來就有 8 台機器」

## 5. 對後續建模與寫作的建議

### 5.1 如果你要引用這個 dataset 本身

建議優先使用以下說法：

- 原始 dataset 是濾材堵塞的 RUL prediction benchmark。
- `Train` 是定期更換前的右設限觀測，`Test` 是帶 RUL 真值的 prediction task。
- 本地 CSV 是扁平化導出版本，沒有保留 `Sampling` 欄位。

### 5.2 如果你要引用 Ding et al. (2025)

建議明確補上以下限定語：

- 該文不是直接使用全部 50 條 `Train` 與 50 條 `Test` 做標準 protocol。
- 它從原始資料中挑選 8 條序列，重映射為 8 台機器。
- 其「機器壽命」對齊的是截尾序列起點的 `RUL`，不是原始完整壽命。
- 其 GRU 輸入特徵、訓練/驗證切分、IM 80% 恢復規則、加噪 augmentation 都是作者自己的建模設計。

### 5.3 如果你要依這篇論文重做實驗

建議在方法章節中額外寫清楚：

- 8 條序列的來源與 `Data_No` 對應表
- 論文壽命值採用的是 `首筆 RUL × 10`
- 是否直接使用 `Test` 標籤作為訓練資料
- 是否重現作者的加噪 augmentation
- 是否保留 `Differential_pressure` 單特徵設定，或加入 `Flow_rate`、`Dust_feed`、`Dust`

## 6. 資料來源

- 原始資料集說明：
  - `Preventive to Predictive Maintenance dataset.pdf`
- 本地 CSV：
  - `Train_Data_CSV.csv`
  - `Test_Data_CSV.csv`
- 對齊論文：
  - `C.Ding_Adaptive_real-time_scheduling_for_production_and_maintenance:Integrating_RUL_prediction_with_multi-agent_deep_reinforcement_learning(2025).pdf`
