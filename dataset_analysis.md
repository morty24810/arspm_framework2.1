# Preventive to Predictive Maintenance Dataset 深度分析與論文對齊

## 1. 分析目的

本報告整合三類證據來源：

1. 原始資料集說明文件 `Preventive to Predictive Maintenance dataset.pdf`
2. 專案根目錄與 `archive/` 目錄中的 CSV，包括 `Train_Data_CSV.csv`、`Test_Data_CSV.csv` 與三個 particle-size raw CSV
3. 論文 `Adaptive real-time scheduling for production and maintenance: Integrating RUL prediction with multi-agent deep reinforcement learning (C. Ding et al., 2025)`

本報告的用途不是重現論文，而是為後續建立較物理化的退化/RUL 模型先整理資料規律。重點包括：

- 原始 dataset 的任務定義與材料背景
- 本地 CSV 與原始 PDF 是否一致
- C. Ding 如何重用這個 dataset
- 粉塵類型、加塵量、流量與堵塞快慢之間可觀察到的規律

## 2. 物理與材料背景

### 2.1 濾材規格與固定條件

原始 PDF 對濾材本體給了明確規格，這對後續物理建模很重要，因為它表示濾材幾何與材料條件基本固定，主要變動來自粉塵、加塵量、流量與製造公差。

| 項目 | 數值 | 來源 |
| --- | --- | --- |
| Mean fibre diameter | `23 μm` | PDF |
| Filter area | `6131 mm²` | PDF |
| Filter thickness | `20 mm` | PDF |
| Filter packing density | `0.014-0.0165` | PDF |
| Clean filter pressure drop | `25 Pa at flow of 1080 m³/(h·m²)` | PDF |
| Filter medium | `CC 600 G` | PDF |

關鍵結論：

- `PDF 直接描述`：資料集中的 measurement samples 都使用同一種濾材 `CC 600 G`。
- `CSV 實證觀察`：本地 CSV 只有粉塵、加塵量、壓差、流量、時間與 RUL，沒有出現多種濾材型號。
- `基於資料的推論`：若要建立物理模型，可以先把濾材規格視為固定條件，把主要變異來源集中在 `Dust`、`Dust_feed`、`Flow_rate` 與不可觀測製造差異上。

### 2.2 粉塵類型、粒徑資訊與密度

原始 PDF 指出，實驗使用標準化 Arizona test dust 的三種粒徑級別：

- `A2 Fine Test Dust`
- `A3 Medium Test Dust`
- `A4 Coarse Test Dust`

PDF 可直接確認的 bulk density 如下：

| 粉塵類型 | 尺寸級別 | Bulk density | 粒徑分布檔 |
| --- | --- | ---: | --- |
| ISO 12103-1 A2 Fine Test Dust | Fine | `0.900 g/cm³` | `Particle size distribution_ISO_12103_1_A2_Fine.mat` |
| ISO 12103-1 A3 Medium Test Dust | Medium | `1.025 g/cm³` | `Particle size distribution_ISO_12103_1_A3_Medium.mat` |
| ISO 12103-1 A4 Coarse Test Dust | Coarse | `1.200 g/cm³` | `Particle size distribution_ISO_12103_1_A4_Coarse.mat` |

關鍵結論：

- `PDF 直接描述`：A2/A3/A4 的粒徑分布不只是類別名稱，還有對應的 distribution function，並另外存成 `.mat` 檔。
- `CSV 實證觀察`：主表 `Train` / `Test` CSV 只保留 `Dust` 名稱，沒有直接展開粒徑分布表。
- `CSV 實證觀察`：`archive/` 下已存在三個 particle-size raw CSV，可直接讀取，且欄位固定為 `SizeUm`, `Channel`, `Passing`。
- `CSV 實證觀察`：三個 raw CSV 都滿足 `Channel.sum() = 100`、`Passing.iloc[-1] = 100`，可視為標準化粒徑分布與累積 passing 曲線。
- `基於資料的推論`：在目前分析階段，particle-size raw CSV 已可作為比 MAT workspace 更高優先級的 primary evidence，後續不需要再依賴 MCOS table 才能取得粒徑分布。

### 2.2.1 粒徑分布量化摘要

下表直接根據新的 particle-size raw CSV 計算，作為後續物理模型最可直接使用的粒徑摘要。

| Dust class | Rows | Size range (`μm`) | first nonzero (`μm`) | last nonzero (`μm`) | `D10` (`μm`) | `D50` (`μm`) | `D90` (`μm`) | 主要高 channel bins (`μm`, `>= 90%` max) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A2 Fine | 64 | `0.0255 - 1408.0` | 0.486 | 176.0 | 1.496 | 8.250 | 42.192 | `3.27 - 31.11` |
| A3 Medium | 64 | `0.0255 - 1408.0` | 0.578 | 248.9 | 2.606 | 14.214 | 59.827 | `7.78 - 44.00` |
| A4 Coarse | 66 | `0.0255 - 2000.0` | 0.818 | 352.0 | 4.467 | 34.620 | 104.761 | `44.00 - 74.00` |

重點摘要：

- `CSV 實證觀察`：`D10`、`D50`、`D90` 都明確呈現 `A2 < A3 < A4`。
- `CSV 實證觀察`：first nonzero size 也呈現 `A2 < A3 < A4`，表示粗粉塵分布整體向較大粒徑右移。
- `基於資料的推論`：A2 Fine 的高 channel bins 分布較寬且偏向小粒徑區間，A4 Coarse 則把主要質量集中在更大的粒徑帶。

### 2.3 Sampling 與時間表示

原始 PDF 的資料結構段明確列出 `Sampling` 欄位，並說明其含義是 measurement data 的記錄頻率（Hz）。另外，用 `scipy` 對 `Data.mat` 做 `varmats_from_mat` 拆分後，可確認檔內確實包含兩個 MATLAB MCOS table：`Test_Data` 與 `Train_Data`。本輪又加入了使用者新轉出的 CSV 證據，可把 sampling 的敘述從「僅確認欄位存在」提升到「固定數值已知」。

目前可確認的事實如下：

- `PDF 直接描述`：`Sampling` 是原始資料結構欄位。
- `CSV 實證觀察`：使用者已明確提供 `Train` / `Test` sampling 都是 `10`。
- `CSV 實證觀察`：`Test` 全部時間差分都等於 `0.1`；`Train` 除 `Data_No=28` 的排序異常外，其餘時間差分眾數也都是 `0.1`。
- `CSV 實證觀察`：`Data.mat` 的 raw workspace 字串可穩定辨識出 `Measured_Data`、`Sampling`、`Dust`、`Dust_feed`、`Data_No`、`Differential_pressure`、`Flow_rate`、`Time`、`RUL` 等欄位標籤。
- `基於資料的推論`：本資料集可視為固定步長的離散時間序列，每步時間解析度為 `0.1`，與 `Sampling = 10 Hz` 完全一致。
- `基於資料的推論`：後文若使用時間斜率、累積加塵 proxy 或壓差變化率，都可以明確視為在 `10 Hz`、`Δt = 0.1` 的固定取樣設定下計算。

## 3. 原始任務與本地 CSV 概況

### 3.1 本地欄位結構

| 檔案 | 欄位 |
| --- | --- |
| `Train_Data_CSV.csv` | `Data_No`, `Differential_pressure`, `Flow_rate`, `Time`, `Dust_feed`, `Dust` |
| `Test_Data_CSV.csv` | `Data_No`, `Differential_pressure`, `Flow_rate`, `Time`, `Dust_feed`, `Dust`, `RUL` |

### 3.2 基本統計

| 檔案 | 總列數 | 序列數 (`Data_No`) | 缺值 | 序列長度 min / median / mean / max | 序列終止時間 min / median / mean / max |
| --- | ---: | ---: | ---: | --- | --- |
| `Train_Data_CSV.csv` | 39,420 | 50 | 0 | 335 / 669.0 / 788.4 / 1794 | 33.5 / 66.9 / 80.79 / 179.4 |
| `Test_Data_CSV.csv` | 39,414 | 50 | 0 | 150 / 558.5 / 788.28 / 2581 | 15.0 / 55.85 / 78.85 / 258.1 |

粉塵類型分布：

| 檔案 | A2 Fine | A3 Medium | A4 Coarse |
| --- | ---: | ---: | ---: |
| `Train_Data_CSV.csv` | 11 | 26 | 13 |
| `Test_Data_CSV.csv` | 14 | 24 | 12 |

### 3.3 原始任務語義驗證

- `PDF 直接描述`：
  - `Train` 是定期更換前的觀測，因此多數序列是 right-censored。
  - 失效條件為 `Differential_pressure > 600 Pa`。
  - `Test` 提供 prediction task 所需的 RUL ground truth。
- `CSV 實證觀察`：
  - `Train` 沒有 `RUL`，`Test` 有 `RUL`。
  - `Train` / `Test` 的固定 sampling 都是 `10`，且 `Time` 的主步長是 `0.1`。
  - `Test` 的 50 條序列都滿足 `Time + RUL = constant`，且 `RUL` 每步固定減 `0.1`。
  - `Train` 中只有 `5/50` 條序列的最大壓差達到或超過 `600 Pa`。
- `基於資料的推論`：
  - 本地 `Train` 很符合「定期更換前截斷觀測」的設計。
  - 本地 `Test` 比較像是「右設限觀測 + 已知剩餘壽命標籤」，而不只是從 0 跑到失效的完整原始歷程。

## 4. 與 C. Ding (2025) 的對齊

### 4.1 關鍵對齊結論

1. 論文確實引用了 Hagmeyer 等人的公開濾材堵塞 RUL dataset。
2. 論文把原始資料中的 `Data_No = 4, 8, 11, 17, 18, 23, 28, 49` 重新映射成 8 台 machine。
3. 這 8 台 machine 的壽命值 `626, 1217, 1267, 1993, 2631, 1285, 2936, 2819`，實際上對齊的是本地 `Test_Data_CSV.csv` 首筆 `RUL × 10`，不是完整生命週期長度。
4. 論文後續只用 machine 5（對應 `Data_No=18`）做 `80% train + 20% validation`，其餘 7 台做 testing，這是作者自行定義的切分方式。

### 4.2 對齊 / 驗證矩陣

| 論文說法 | 原始 PDF 是否支持 | 本地 CSV 是否支持 | 結論標籤 | 說明 |
| --- | --- | --- | --- | --- |
| 觀測變數包含 `flow rate` 與 `differential pressure` | 是 | 是 | `已驗證` | 原始 PDF 與本地 CSV 一致。 |
| dataset 的目標是預測 RUL，以支援 predictive maintenance | 是 | 是 | `已驗證` | 任務描述與 `Test` 標籤一致。 |
| 使用 `Data_No = 4, 8, 11, 17, 18, 23, 28, 49` 代表 machine 1~8 | 否 | 部分支持 | `部分驗證` | 本地 CSV 證明這些序列存在，但「8 台機器」是作者重建的任務抽象。 |
| 8 台 machine 的 lifespan 為 `626, 1217, 1267, 1993, 2631, 1285, 2936, 2819` | 否 | 部分支持 | `表述不精確` | 它對齊的是 `Test` 首筆 `RUL × 10`，不是完整總壽命。 |
| 這 8 條序列具有 complete operating lifecycle information | 部分支持 | 否 | `表述不精確` | 本地 `Test` 尾端 `RUL` 並未降到 0。 |
| `Differential_pressure` 是 RUL 模型唯一輸入 | 否 | 是 | `已驗證` | 這是作者特徵選擇，不是 dataset 的原始限制。 |
| machine 5 用於 train/validation，其餘 7 台做 testing | 否 | 否 | `無法直接驗證` | 這是作者自定 split。 |
| IM 把壽命恢復到前次剩餘壽命的 80%，CM 完全恢復 | 否 | 否 | `無法直接驗證` | 屬於論文模擬規則。 |
| 使用 noise augmentation 模擬即時數據 | 否 | 否 | `無法直接驗證` | 屬於作者後處理。 |
| 原始結構包含 `Sampling` | 是 | 部分支持 | `已驗證` | 主表 CSV 未保留欄位，但使用者新轉出的證據與 `Time=0.1` 步長共同支持 `10 Hz`。 |

## 5. 論文所選 8 條序列的材料與退化特徵

下表專門把 C. Ding 選用的 8 條序列整理成後續物理建模可直接引用的摘要。`累積加塵 proxy` 與 `累積加塵質量 proxy` 都只作相對比較，不作嚴格單位校驗。

| Machine | `Data_No` | Dust | Size class | `Dust_feed` | Bulk density | 首筆 `RUL` | `Time + RUL` | 平均 `Flow_rate` | 最大 `DP` | 末端 `DP` | `DP-Time` 線性斜率 proxy | 累積加塵 proxy | 累積質量 proxy |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 4 | ISO 12103-1 A3 Medium | Medium | 316.985065 | 1.025 | 62.6 | 62.8 | 55.97 | 540.5 | 540.5 | 9.071 | 19906.7 | 20404.3 |
| M2 | 8 | ISO 12103-1 A3 Medium | Medium | 158.492533 | 1.025 | 121.7 | 121.8 | 58.10 | 552.8 | 552.8 | 5.268 | 19304.4 | 19787.0 |
| M3 | 11 | ISO 12103-1 A3 Medium | Medium | 158.492533 | 1.025 | 126.7 | 126.8 | 81.70 | 503.8 | 501.9 | 4.906 | 20096.9 | 20599.3 |
| M4 | 17 | ISO 12103-1 A3 Medium | Medium | 59.107236 | 1.025 | 199.3 | 199.4 | 81.58 | 575.0 | 560.3 | 3.123 | 11786.0 | 12080.6 |
| M5 | 18 | ISO 12103-1 A3 Medium | Medium | 59.107236 | 1.025 | 263.1 | 263.2 | 81.74 | 562.7 | 550.1 | 2.174 | 15557.0 | 15946.0 |
| M6 | 23 | ISO 12103-1 A3 Medium | Medium | 118.214472 | 1.025 | 128.5 | 128.6 | 59.30 | 470.6 | 468.8 | 3.664 | 15202.4 | 15582.4 |
| M7 | 28 | ISO 12103-1 A4 Coarse | Coarse | 59.107236 | 1.200 | 293.6 | 293.7 | 82.80 | 398.9 | 397.7 | 1.546 | 17359.8 | 20831.8 |
| M8 | 49 | ISO 12103-1 A4 Coarse | Coarse | 59.107236 | 1.200 | 281.9 | 282.0 | 80.39 | 404.3 | 404.3 | 1.507 | 16668.2 | 20001.9 |

以下粒徑指標按 dust class 映射到 8 條序列，不對單條序列重複估計：

| Machine | `Data_No` | Dust class | first nonzero (`μm`) | `D10` (`μm`) | `D50` (`μm`) | `D90` (`μm`) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| M1 | 4 | A3 Medium | 0.578 | 2.606 | 14.214 | 59.827 |
| M2 | 8 | A3 Medium | 0.578 | 2.606 | 14.214 | 59.827 |
| M3 | 11 | A3 Medium | 0.578 | 2.606 | 14.214 | 59.827 |
| M4 | 17 | A3 Medium | 0.578 | 2.606 | 14.214 | 59.827 |
| M5 | 18 | A3 Medium | 0.578 | 2.606 | 14.214 | 59.827 |
| M6 | 23 | A3 Medium | 0.578 | 2.606 | 14.214 | 59.827 |
| M7 | 28 | A4 Coarse | 0.818 | 4.467 | 34.620 | 104.761 |
| M8 | 49 | A4 Coarse | 0.818 | 4.467 | 34.620 | 104.761 |

重點摘要：

- `CSV 實證觀察`：這 8 條序列中，`6` 條是 A3 Medium，`2` 條是 A4 Coarse，`0` 條是 A2 Fine。
- `CSV 實證觀察`：論文的 training machine 5 對應 `Data_No=18`，屬於 `A3 Medium`，且 `Dust_feed=59.107236`，屬於低 feed 組。
- `基於資料的推論`：C. Ding 的 RUL 模型沒有接觸 A2 Fine，也沒有覆蓋完整的 dust-type 組合，因此它對粉塵尺寸效應的泛化能力先天受限。

## 6. 粉塵尺寸與堵塞快慢的資料規律

### 6.1 先看粒徑分布本身

若只看新的 particle-size raw CSV，而不先看壽命，A2/A3/A4 已經形成非常清楚的粒徑階層：

| Dust class | `D10` (`μm`) | `D50` (`μm`) | `D90` (`μm`) | first nonzero (`μm`) | 高 channel bins (`μm`) |
| --- | ---: | ---: | ---: | ---: | --- |
| A2 Fine | 1.496 | 8.250 | 42.192 | 0.486 | `3.27 - 31.11` |
| A3 Medium | 2.606 | 14.214 | 59.827 | 0.578 | `7.78 - 44.00` |
| A4 Coarse | 4.467 | 34.620 | 104.761 | 0.818 | `44.00 - 74.00` |

解讀：

- `CSV 實證觀察`：`D10`、`D50`、`D90` 的排序都一致，明確支持 `A2 < A3 < A4`。
- `CSV 實證觀察`：A4 Coarse 的主要高 channel bins 已右移到 `44 - 74 μm`，與 A2 Fine 的 `3.27 - 31.11 μm` 形成明顯區隔。
- `基於資料的推論`：這組粒徑指標提供了比單純 `Dust` 類別名稱更可量化的材料差異，可直接拿來跟後面的壽命與壓差統計對齊。

### 6.2 全資料集層：不同粉塵類型的壽命分布

以下統計使用整個 `Test_Data_CSV.csv`。

| Dust class | 序列數 | 平均總壽命 (`Time + RUL`) | 中位數總壽命 | 最小總壽命 | 最大總壽命 | 平均 `Dust_feed` | 平均 `Flow_rate` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 Fine | 14 | 65.67 | 44.70 | 26.2 | 202.4 | 215.97 | 72.07 |
| A3 Medium | 24 | 119.49 | 103.95 | 49.2 | 263.2 | 168.96 | 63.57 |
| A4 Coarse | 12 | 166.87 | 119.45 | 47.2 | 335.2 | 156.49 | 81.08 |

解讀：

- `CSV 實證觀察`：全體平均壽命排序 `A4 Coarse > A3 Medium > A2 Fine`，與 `D50` / `D90` 的粒徑排序方向一致。
- `基於資料的推論`：較細的粉塵傾向更快造成壓差上升與壽命縮短，但這不是純粒徑單因子結果，因為不同粉塵組的 feed 與 flow 分布也不相同。

### 6.3 同一 `Dust_feed` 下的壽命比較

若把 `Dust_feed` 固定，粉塵尺寸帶來的差異更容易觀察。

| `Dust_feed` | A2 Fine 平均總壽命 | A3 Medium 平均總壽命 | A4 Coarse 平均總壽命 |
| ---: | ---: | ---: | ---: |
| 59.107236 | 195.80 | 231.30 | 303.48 |
| 118.214472 | - | 125.67 | 145.50 |
| 158.492533 | - | 102.07 | 93.40 |
| 236.428943 | 37.07 | 66.65 | 64.20 |
| 237.738799 | 42.60 | 82.70 | 70.20 |
| 316.985065 | 42.70 | 57.33 | 53.70 |

補充判讀：

- `CSV 實證觀察`：在 `59.107236` 這個低 feed 組，壽命呈現 `A4 > A3 > A2`，差異非常明顯。
- `CSV 實證觀察`：在高 feed 組 `236.428943`、`237.738799`、`316.985065`，A2 Fine 仍然是最短壽命組。
- `基於資料的推論`：把這個結果與 `D10/D50/D90` 排序放在一起看，資料支持的是「更細的粒徑分布通常對應更快堵塞」；但 A3 與 A4 的相對順序在部分 feed 組仍會受流量與個別序列差異影響，因此不宜把「粗粉塵一定最慢堵塞」寫成絕對定律。

### 6.4 壓差跨越門檻的時間

以下使用每條序列第一次跨越某個壓差門檻的 `Time` 作比較。

| Dust class | 100 Pa 平均時間 | 200 Pa 平均時間 | 300 Pa 平均時間 | 400 Pa 平均時間 |
| --- | ---: | ---: | ---: | ---: |
| A2 Fine | 30.52 | 47.03 | 36.43 | 43.60 |
| A3 Medium | 53.57 | 72.66 | 91.04 | 107.76 |
| A4 Coarse | 95.15 | 126.51 | 142.22 | 147.70 |

補充判讀：

- `CSV 實證觀察`：A2 Fine 最快達到 100/200/400 Pa 門檻，A4 Coarse 最慢。
- `基於資料的推論`：若把壓差門檻視為堵塞進程的 proxy，這個排序與 `D10/D50/D90` 的粒徑排序一致，支持「細粉塵較快促進堵塞，粗粉塵較慢」這個方向。
- `基於資料的推論`：300/400 Pa 的統計樣本數在 A2、A4 類別較少，因此高門檻區間的比較要保守解讀。

### 6.5 C. Ding 的 8 條子集層

這 8 條子集與全資料集相比，有三個明顯偏差：

1. 完全沒有 `A2 Fine`
2. A3/A4 的比例是 `6:2`
3. training machine 5 對應 `Data_No=18`，屬於 A3 Medium、低 feed、較長壽命

這代表：

- `CSV 實證觀察`：論文子集沒有覆蓋最容易快速堵塞的 A2 Fine 區間。
- `基於資料的推論`：若直接沿用 C. Ding 的子集建 RUL 模型，模型很可能更像是在學 A3/A4 的退化規律，而不是整個 dataset 的完整粉塵行為空間。
- `基於資料的推論`：若你想做更真實的物理模型，應該把 `Dust`、`Dust_feed`、`Flow_rate` 與 loading proxy 視為顯性輸入或顯性條件，而不是只把壓差序列做黑箱正規化。

## 7. 對物理模型建構的啟示

這一節只整理資料規律，不提出具體模型形式。

### 7.1 可直接作為先驗的事實

- `PDF 直接描述`：濾材規格固定，且測試樣本都使用 `CC 600 G`。
- `PDF 直接描述`：training data 的 censoring time 會隨 particle feed 近似反比縮放。
- `PDF 直接描述`：壽命變異主要來自 `type of dust`、`flow rate` 與 manufacturing tolerances。

### 7.2 可直接利用的資料規律

- `CSV 實證觀察`：`Dust_feed` 在每條序列內固定，且只落在少數離散檔位。
- `CSV 實證觀察`：particle-size raw CSV 已可直接提供 `SizeUm / Channel / Passing`，而不再只是 `Dust` 類別名稱。
- `CSV 實證觀察`：`Time + RUL` 對 `Test` 每條序列為常數，可視為該截尾序列對應的總壽命標記。
- `CSV 實證觀察`：A2 Fine 普遍最短壽命，A4 Coarse 普遍最長。
- `CSV 實證觀察`：相同 feed 下，不同 dust 類別的總壽命差異仍然存在。

### 7.3 對物理模型更直接有用的粒徑特徵

在目前資料條件下，若你要從「純 RUL 黑箱」往更物理化的模型前進，最直接可用的輸入不是再做更多 heuristic，而是把下列量顯性化：

- `CSV 實證觀察`：`A2/A3/A4` 類別 one-hot。
- `CSV 實證觀察`：`D10`、`D50`、`D90`。
- `CSV 實證觀察`：完整 cumulative passing curve（A2/A3 為 64 bins，A4 為 66 bins）。
- `CSV 實證觀察`：`Dust_feed`。
- `CSV 實證觀察`：`Flow_rate`。
- `CSV 實證觀察`：`Sampling = 10` 與固定 `Δt = 0.1` 的時間離散設定。

這些特徵的角色可以理解為：

- `基於資料的推論`：`Dust` 類別提供粗粒度材料標籤。
- `基於資料的推論`：`D10/D50/D90` 與完整 passing curve 提供更連續的粒徑描述。
- `基於資料的推論`：`Dust_feed`、`Flow_rate` 與 `Sampling=10` 共同決定每個時間步上的外部負荷與退化解析度。

### 7.4 對建模的實際含義

- `基於資料的推論`：純時間不是最物理的退化自變數；`Dust_feed × total_life` 與再乘上 bulk density 的 loading proxy，更接近濾材累積負荷。
- `基於資料的推論`：若把粒徑分布也納入，`Dust_feed × total_life` 最好被理解為總 loading proxy，而不是已吸收全部材料效應的充分統計量。
- `基於資料的推論`：若只用 `Differential_pressure` 做單特徵 RUL 預測，會把 `Dust`、`Flow_rate`、`Dust_feed` 對堵塞機理的影響全部隱含到黑箱裡。
- `基於資料的推論`：C. Ding 的單特徵、單機訓練、8 序列子集做法可以當 baseline，但不適合作為最終物理模型的完整依據。

## 8. 資料品質與使用注意事項

### 8.1 `Train` 的排序異常

`Train_Data_CSV.csv` 的 `Data_No=28` 有明顯時間排序異常：

- 首筆時間是 `157.2`
- 接著回到 `0.2, 0.3, 0.4 ...`

因此：

- `CSV 實證觀察`：`Train` 中只有 `49/50` 條序列保持嚴格時間遞增。
- `基於資料的推論`：若模型依賴序列順序，必須先對 `Train` 做排序或清洗，至少要單獨處理 `Data_No=28`。

### 8.2 關於 MAT 原始檔

目前本地環境可確認：

- `Data.mat` 存在，且可拆出兩個 MATLAB MCOS table：`Test_Data` 與 `Train_Data`
- `Train_Data_Uncensored.mat` 另以 `Train_Data_Uncensored` 的 MATLAB MCOS table 形式存在
- `archive/` 下已存在三個可直接讀取的 particle-size raw CSV，欄位為 `SizeUm`, `Channel`, `Passing`
- 三個 particle-size `.mat` 檔存在，且可辨識為 `raw` table 類型
- `Data.mat` raw workspace 可辨識出 `Measured_Data`、`Sampling`、`Dust`、`Dust_feed`、`Data_No`、`Differential_pressure`、`Flow_rate`、`Time`、`RUL`
- particle-size raw workspace 可辨識出 `SizeUm`、`Size um`、`Channel`、`Passing`、`Description`、`VariableUnits`、`VariableDescriptions`

這表示：

- 本報告已可直接使用新的 raw CSV 驗證 A2/A3/A4 的完整粒徑分布
- `Sampling = 10` 已由使用者轉檔資訊與 `Time = 0.1` 步長共同驗證
- MATLAB MCOS table 的限制，已不再阻礙本輪的粒徑與 sampling 分析

### 8.3 MATLAB Engine 嘗試結果與阻塞點

為了進一步驗證 `Sampling` 與 particle-size `.mat` 的原始欄位，本輪另外嘗試使用本機 MATLAB R2024b 的官方 Python engine。

目前可確認的事實如下：

- `/Applications/MATLAB_R2024b.app` 存在，且其 `extern/engines/python/setup.py` 明確支援 `Python 3.9-3.12`。
- 在目前環境中，直接用 `python -m pip install /Applications/MATLAB_R2024b.app/extern/engines/python` 無法完成安裝：
  - build isolation 會因受限網路而無法額外抓取 `setuptools`
  - 關閉 build isolation 後，wheel build 又會嘗試在 `/Applications/...` 下建立 `build/`，並因權限限制失敗
- 不經 pip 安裝時，可透過設定 `MWE_INSTALL` 與 `PYTHONPATH` 直接成功 `import matlab` 與 `import matlab.engine`。
- 但在目前環境下，`matlab.engine.start_matlab()` 與 `matlab.engine.start_matlab('-batch \"disp(1)\"')` 都穩定回傳 `EngineError: Transport stopped`。
- 直接呼叫 `/Applications/MATLAB_R2024b.app/bin/matlab -batch "disp(1)"` 也只得到 exit code `1`，沒有可用的標準輸出錯誤訊息。

因此本輪結論是：

- `CSV 實證觀察`：MATLAB Engine 套件本身可在 Python 端被載入。
- `基於資料的推論`：阻塞點出在 MATLAB process/engine transport 啟動，而不是純 Python import。
- `基於資料的推論`：在新的 raw CSV 已可直接使用後，MATLAB Engine 失敗不再影響本輪對 `Sampling=10` 與 particle-size distribution 的分析結論；它現在只影響未來是否還要回頭驗證 MAT 內部 object 表示。

### 8.4 `pymatreader` 與 Octave 替代路徑結果

在 MATLAB Engine 之外，本輪也另外嘗試了 `pymatreader` 與 GNU Octave：

- `pymatreader 1.2.2` 已成功安裝到隔離路徑 `/tmp/pymatreader_local`
- 本機環境中 `octave` / `octave-cli` 不在 PATH，且在常見安裝位置下未找到可執行檔

實際結果如下：

- `CSV 實證觀察`：`pymatreader` 與 `scipy.loadmat` 的結果一致，仍將這些 MAT 識別為 MATLAB `MCOS table`，並暴露 `__function_workspace__`。
- `CSV 實證觀察`：對 `Data.mat` 使用 `varmats_from_mat` 可把兩個 opaque table 分離為 `Test_Data` 與 `Train_Data`。
- `CSV 實證觀察`：對 particle-size MAT 的 workspace 字串抽取，可穩定看到 `SizeUm` / `Size um`、`Channel`、`Passing` 等標籤。
- `基於資料的推論`：在 raw CSV 已存在後，這些替代路徑的主要價值變成驗證 MAT 結構，而不是提供本輪粒徑分析所需的 primary evidence。

## 9. 資料來源

- 原始資料集說明：`Preventive to Predictive Maintenance dataset.pdf`
- 本地資料：`Train_Data_CSV.csv`、`Test_Data_CSV.csv`
- 原始 archive 工件：`Data.mat`、`Train_Data_Uncensored.mat`、三個 `Particle size distribution_*.mat`
- 新轉出的原始分布 CSV：三個 `Particle size distribution_*_raw.csv`
- 對齊論文：`C.Ding_Adaptive_real-time_scheduling_for_production_and_maintenance:Integrating_RUL_prediction_with_multi-agent_deep_reinforcement_learning(2025).pdf`
