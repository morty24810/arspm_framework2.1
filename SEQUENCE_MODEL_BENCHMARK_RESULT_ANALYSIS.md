# 時序模型 Benchmark 結果分析

## 1. 分析範圍

本文件整合兩次正式 benchmark run，並以資料集結構一起解讀結果：

- 上一版正式基線：
  - `outputs/sequence_model_benchmark/20260313_141821`
- 本次優化 run：
  - `outputs/sequence_model_benchmark/20260313_181038`

明確排除以下未完成 run：

- `outputs/sequence_model_benchmark_v2/20260313_135556`

因為該 run 沒有完整跑完所有 family / model，也沒有輸出最終 aggregate metrics。

本分析同時引用四類證據：

1. 兩次 benchmark run 的 `metrics_overall.csv`、`metrics_by_family.csv`、`metrics_by_fold.csv`
2. 本次 run 的 `acceptance_summary.json`、`baseline_compare_phase1.csv`、`baseline_compare_phase2.csv`、`protocol_vs_tuned_compare.csv`
3. 原始 `Train_Data_CSV.csv` 與 `Test_Data_CSV.csv`
4. 既有資料集研究文件 [dataset_analysis.md](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/dataset_analysis.md)

## 2. 上一版正式基線分析摘要

上一版基線 run `20260313_141821` 的主要結論仍然成立，這一版必須保留作為對照基準。

### 2.1 整體結果

來自 `metrics_overall.csv` 的整體模型排序如下：

| 排名 | 模型 | Health RMSE | Health R2 | RUL RMSE |
| --- | --- | ---: | ---: | ---: |
| 1 | `LSTM` | `0.0638` | `0.8951` | `8.7127` |
| 2 | `GRU` | `0.0682` | `0.8792` | `9.2312` |
| 3 | `ATTENTION` | `0.0689` | `0.8650` | `9.5683` |
| 4 | `TCN` | `0.0713` | `0.8564` | `10.0200` |

這一版的核心結論是：

- `health_remaining` 已經不再像上一代方法那樣只是 `DP` 的線性重標定。
- `RUL` 也不再有 slope inversion 帶來的大幅 bouncing。
- `LSTM` 是當時最合理的預設模型。

### 2.2 Family 層級表現

上一版 `LSTM` 的 family 指標如下：

| Family | Health RMSE | Health R2 | RUL RMSE |
| --- | ---: | ---: | ---: |
| `A2` | `0.0874` | `0.7945` | `9.0349` |
| `A3` | `0.0703` | `0.9170` | `11.0410` |
| `A4` | `0.0336` | `0.9739` | `6.0623` |

當時的解讀是：

- `A4` 最穩，代表新 target semantics 在粗粉塵族群上對齊最好。
- `A3` 居中，說明它不像 `A2` 有明顯 train/test feed shift，但族內退化模式仍較複雜。
- `A2` 最難，且主要問題不是單純高 feed extrapolation，而是低 feed 長壽命稀有子群。

### 2.3 上一版的主要問題

上一版已明確指出四個未解決問題：

1. `A2` 的低 feed 長壽命稀有子群仍然被系統性低估。
2. 曲線不再大幅 bouncing，但局部鋸齒仍存在。
3. benchmark 內的 `RUL` 依賴真實 `total_life`，因此還不能直接視為 simulator inference。
4. 很多 `Test` 序列沒有實際觀測到 failure，圖上的 true tail 是 reference，不是模型真的看到的尾段。

## 3. 本次優化 Run 的整體結果

本次優化 run `20260313_181038` 的 acceptance 結論來自 [acceptance_summary.json](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/acceptance_summary.json)：

- `phase1_protocol.accepted_as_default = false`
- `phase2_tuned.accepted_as_default = false`

也就是說，這次優化沒有通過「拉升 `A3`、同時不能讓 `A2/A4` 退化」的驗收門檻。

### 3.1 整體指標對照

| Run | Health RMSE | Health R2 | RUL RMSE |
| --- | ---: | ---: | ---: |
| 基線 `20260313_141821` | `0.0638` | `0.8951` | `8.7127` |
| 本次 `20260313_181038` | `0.0703` | `0.8829` | `9.5557` |
| 差值（本次 - 基線） | `+0.0066` | `-0.0123` | `+0.8430` |

結論：

- 這次 run 在 overall 層級是退步的。
- 退步不是只發生在單一 family，而是至少 `A3` 與 `A4` 都一起變差。
- 因此不能把這次視為「A3 提升、其他持平」的成功版本。

### 3.2 Family 指標對照

| Family | Baseline Health RMSE | Current Health RMSE | Baseline RUL RMSE | Current RUL RMSE | 解讀 |
| --- | ---: | ---: | ---: | ---: | --- |
| `A2` | `0.0874` | `0.0887` | `9.0349` | `7.9872` | `health` 微退，但 `RUL` 明顯改善 |
| `A3` | `0.0703` | `0.0859` | `11.0410` | `13.3669` | 兩項都退化，是本次主失敗點 |
| `A4` | `0.0336` | `0.0364` | `6.0623` | `7.3131` | 兩項都退化，不能接受 |

直接結論：

- 這次不是全面失敗，而是「A2 的某些補丁有效，但 A3/A4 被一起傷到」。
- 若只看 `A2`，新 protocol 確實有價值。
- 但如果看整體預設配置，它還不能取代基線。

## 4. 與資料集結構一起解讀

這次結果如果不結合 dataset 結構，很容易誤判成「模型不夠強」；但實際上更大的問題是 family 內部結構差異。

### 4.1 A2 的問題仍然符合資料集結構

本次 run 的 [dataset_summary.json](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/dataset_summary.json) 再次確認：

- `A2` train feed：`59.107235, 79.246269, 118.214470, 158.492538`
- `A2` test feed：`59.107235, 177.321701, 236.428940, 237.738800, 316.985077`
- overlap 只有 `59.107235`

但 `A2` 真正最難的樣本不是高 feed，而是低 feed 長壽命稀有子群：

- `Data_No=19`：`feed=59.107236`，`total_life=348.0`
- `Data_No=20`：`feed=59.107236`，`total_life=261.0`

這解釋了為什麼本次 [rul_curve_A2_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/rul_curve_A2_LSTM.png) 仍然是系統性低估，但 worst-fold `RUL RMSE` 已經從基線的 `35.64` 改善到 `28.67`。

所以 `A2` 的結論和上一版一致：

- 問題不是單純 feed coverage 不足。
- 問題是稀有低 feed 長壽命子群在 grouped fold 下非常脆弱。
- 這次 `A2` 補償策略有一定效果，但還沒把 `health` 一起拉穩。

### 4.2 A3 的問題不是資料量不足，而是族內異質性太高

如果只看 sequence 數量，`A3` 並不弱：

- `Train`：`26` 條
- `Test`：`24` 條

而且和 `A2` 不同，`A3` 的 train/test feed coverage 是完整對齊的，`8` 個 feed 都同時出現在 train 和 test。

但把 `A3` 按 feed、flow、壽命一起看，會發現它的內部結構很雜：

- `59.107236`：`flow_mean ≈ 81.6`，`total_life 392.7 - 521.0`
- `79.246266`：`flow_mean ≈ 57.9`，`total_life 313.9 - 345.2`
- `158.492533`：`flow_mean 57.8 - 81.7`，`total_life 121.1 - 236.2`

也就是說，`A3` 的困難不只是 feed 多，而是：

- 同一個 family 內混了不同壽命層級
- 同一個 feed 內還可能混了不同 flow regime
- 這會讓單一 shared protocol 更容易把生命進度壓成平均形狀

這也是為什麼 [health_curve_A3_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/health_curve_A3_LSTM.png) 和 [rul_curve_A3_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/rul_curve_A3_LSTM.png) 雖然整體方向對，但會在中後段整體偏高估。

### 4.3 A4 的穩定性來自更乾淨的家族結構

根據 [dataset_analysis.md](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/dataset_analysis.md)：

- `A4` 是粒徑最粗的族群，`D50 = 34.620`
- `A4` 的平均壽命也是三族中最長

本地 CSV 再往下看，`A4` 的 family 內部也比 `A3` 乾淨很多：

- 多數 feed 的 `flow_mean` 都穩定在 `80 - 82`
- `feed` 與壽命的關係更一致
- 長壽命樣本主要集中在低 feed，而不是同一 feed 下分裂成多種 regime

這和 [rul_curve_A4_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/rul_curve_A4_LSTM.png) 的表現一致：

- 曲線很順
- 生命週期方向是對的
- 只是本次優化版本比基線更保守，因此指標反而變差

### 4.4 很多 Test 序列並沒有觀測到 failure

`dataset_summary.json` 也再次提醒：

- `A2` terminal `health_remaining` 最大到 `0.6200`
- `A3` 最大到 `0.6402`
- `A4` 最大到 `0.5299`

因此圖上的 dotted `observed end` 之後：

- true line 只是 reference tail
- 不是觀測到的真實尾段
- 也不是模型真正預測出來的後續軌跡

這一點在解釋圖時必須持續保留，不然會把 benchmark 的可視化讀得過度樂觀。

## 5. 圖表與數據對照解讀

### 5.1 Health scatter

見 [scatter_health_remaining_pred.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/scatter_health_remaining_pred.png)

這張圖的訊息是：

- 點雲仍然圍繞對角線，代表生命進度語義沒有跑掉。
- 高 `health` 區間出現明顯壓縮與分層，說明 monotone projection 讓預測更穩，但也更容易形成 plateau。
- 它不像上一代 slope-based 方法那樣失真，但也沒有比基線更準。

### 5.2 RUL scatter

見 [scatter_rul_pred.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/scatter_rul_pred.png)

這張圖說明：

- 大幅 bouncing 確實已經不見了。
- 但在 `true RUL > 200` 的區域仍有系統性低估。
- 這個低估帶和 `A2` / `A3` / `A4` 的低 feed 長壽命序列完全對得上。

所以 `RUL` 的穩定性是進步了，但高壽命區間的偏差還沒有真正解掉。

### 5.3 A2 代表曲線

見 [rul_curve_A2_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/rul_curve_A2_LSTM.png)

圖上表現：

- 不再劇烈上跳或爆炸。
- 但整段仍低於 true line。
- 偏差主要集中在長壽命低 feed 行為。

這和資料集中的 `A2 Data_No=19,20` 完全一致。

### 5.4 A3 代表曲線

見 [health_curve_A3_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/health_curve_A3_LSTM.png) 與 [rul_curve_A3_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/rul_curve_A3_LSTM.png)

圖上表現：

- 整體方向正確，證明 target semantics 仍是對的。
- 預測曲線明顯比 true line 更保守、更高。
- 局部階梯狀的下降比基線更明顯。

這更像是：

- 新加權策略和 monotone projection 沒有讓 A3 真正對齊。
- 反而把 A3 的族內異質性壓成了過度平滑的平均曲線。

### 5.5 A4 代表曲線

見 [rul_curve_A4_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_181038/plots/rul_curve_A4_LSTM.png)

圖上表現：

- 曲線仍然非常平順。
- 生命進度方向正確。
- 但相較基線，這次的預測偏得更保守，因此 `RUL RMSE` 反而變差。

也就是說，`A4` 不是學壞了，而是被這次 protocol 一起拖退了。

## 6. 本次結果是否符合預期

需要分兩層回答。

### 6.1 如果問題是「生命進度語義是否仍然對齊」

答案是：

- 是，仍然對齊。

理由：

- `health_remaining` 沒有退回去變成 `DP proxy`
- `RUL` 也沒有重現舊版的爆炸式 bouncing
- 圖上仍然能看出 family 與壽命層級的合理對應

### 6.2 如果問題是「這次優化是否成功拉升 A3，且不讓 A2/A4 退化」

答案是：

- 否，沒有達成。

理由：

- `A3` 沒有變好，反而是本次最大的退化來源。
- `A4` 也一起退化。
- `A2` 的 `RUL` 雖然改善，但 `health` 沒守住基線。
- 驗收檔已經明確判定不可作為新的 default baseline。

## 7. 這次暴露出的主要問題

1. `A2` 的主要難點仍然是低 feed 長壽命稀有子群，這次只壓低了 worst-fold `RUL`，還沒有真正把 `health` 對齊。

2. `A3` 的主問題不是資料量不足，而是同一 family 內混有多個 `feed x flow x life` regime，單一 shared protocol 很容易學成平均曲線。

3. monotone projection 雖然解決了大 bouncing，但也讓高 health 區間更容易出現 plateau 與壓縮，可能正在傷害 `A3/A4` 的 RMSE。

4. family tuning search 的最佳結果，沒有在最終確認 run 中穩定重現。以 `A3` 為例，搜尋階段最佳配置曾達到：
   - `health_rmse = 0.0690`
   - `rul_rmse = 10.8622`
   但最終正式 run 同配置卻掉到：
   - `health_rmse = 0.0859`
   - `rul_rmse = 13.3669`

5. benchmark 內的 `RUL` 仍依賴真實 `total_life`，因此這次分析能證明 target semantics 與 family 結構有對上，但還不能直接等同 simulator 端的推理品質。

## 8. 下一步改良方向

下一步不應該再直接大幅加模型複雜度，而應該先把問題拆清楚。

### 8.1 先解決 tuning 不可重現

這是目前最急迫的工程問題。

應先驗證：

- 搜尋階段與最終確認 run 的 seed、資料切分、early stopping 是否完全一致
- 同一配置重跑多次時，指標波動有多大
- 目前的 best-config 選擇是否被訓練方差誤導

如果這一點不先修好，後續任何 `A3` 改良都可能只是偶然好結果。

### 8.2 對 A3 做明確的結構化分群分析

`A3` 現在看起來最需要的不是更多 data，而是更明確地辨識族內 regime。

下一步應該先驗證：

- `A3` 在 `feed x flow` 空間是否存在穩定子群
- 這些子群和 `total_life` 是否有可分離的對應關係
- 目前錯誤最大的 fold，是否正好把某一類 regime 整塊 hold out

### 8.3 保留 A2 補丁，但不要再讓它拖累 A3/A4

`A2` 的長壽命補償是有價值的，因為它確實改善了 worst-fold `RUL`。

但下一步要驗證：

- 這個補償是否應只作用在 `A2`
- 它是否透過 shared training protocol 間接改壞了 `A3/A4`
- `A2` 的補償係數是否應更細分，而不是單一倍數

### 8.4 把 raw 與 projected 指標正式分開看

目前官方 summary 用的是 projected prediction，這對語義上是合理的，但不利於判斷：

- 問題到底來自模型本身
- 還是來自 monotone projection 的後處理副作用

下一輪建議同時保留：

- raw metrics
- projected metrics

並分開比較 `A3/A4` 是否是被 monotone projection 拉差。

### 8.5 維持 benchmark 與 simulator 的邊界

現階段比較合理的做法是：

- 先把 benchmark 裡的生命進度學習與 family-level 結構對齊好
- 等到 `A3` 真的穩定拉上來、而且 tuning 可重現後
- 再考慮接回 simulator / RL 主流程

## 9. 總結

如果只看方向，這次 run 仍然支持 v2 的核心判斷：

- `health` 應該定義為生命進度，而不是 `DP proxy`
- `RUL` 的穩定改善主要來自 target semantics 對齊，而不是單純模型更強

但如果看這一輪優化的實際目標，結論是：

- 這次沒有成功把 `A3` 拉近 `A4`
- 反而讓 `A3/A4` 一起退化
- `A2` 雖有局部改善，但還不足以支持切換到新 baseline

所以目前最合理的立場是：

- 保留 `20260313_141821` 作為鎖定基線
- 把 `20260313_181038` 視為一次有價值但未通過驗收的優化嘗試
- 下一步優先處理 `A3` 的族內異質性與 tuning 不可重現問題
