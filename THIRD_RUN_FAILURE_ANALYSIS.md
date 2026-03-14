# 第三輪 Benchmark 效果惡化原因說明

## 背景

這份文件用來記錄第三輪自動化 benchmark 為什麼會比前兩輪差很多，避免之後重複走同一條錯誤路徑。

第三輪指的是在前兩輪基線與優化 run 之後，加入以下兩類改動的版本：

- strict reproducibility 機制
- prefix-derived sequence context 路徑

## 結論摘要

第三輪變差的主因，不是資料集突然變難，也不是單純 `A3` 比 `A4` 更難學，而是新版本訓練流程本身出現了回歸。

更準確地說：

- 可重現性問題其實有改善
- 但模型訓練在新架構下沒有正常收斂
- 因此最終指標全面崩潰

所以第三輪的失敗，本質上是一次工程回歸，不是一個可以拿來比較建模能力的有效 run。

## 直接觀察到的異常現象

第三輪的整體表現大幅差於前兩輪：

| Run | Health RMSE | Health R2 | RUL RMSE |
| --- | ---: | ---: | ---: |
| 第一輪正式基線 | `0.0638` | `0.8951` | `8.7127` |
| 第二輪優化 run | `0.0703` | `0.8829` | `9.5557` |
| 第三輪自動化改良 run | `0.6381` | `-8.1947` | `104.5585` |

而且不是只有某一個 family 失敗，而是三個 family 一起崩掉：

- `A2`: `health_rmse = 0.6149`, `rul_rmse = 59.3508`
- `A3`: `health_rmse = 0.6599`, `rul_rmse = 94.9893`
- `A4`: `health_rmse = 0.6395`, `rul_rmse = 159.3355`

這種全家族同步惡化的模式，不符合資料分布差異造成的局部退化，更像訓練程序本身出錯。

## 為什麼判斷是訓練流程回歸

第三輪最關鍵的訊號，是每個 fold 都出現相同的異常：

- `pretrain_best_epoch = 0`
- `pretrain_best_rmse = inf`
- `best_epoch = 0`
- `best_val_rmse = inf`

這說明至少有一件事成立：

1. pretrain 階段沒有產生可接受的最佳驗證結果。
2. finetune 階段也沒有產生可接受的最佳驗證結果。
3. 問題不是「收斂得不好」，而是「最佳模型追蹤機制從頭到尾都沒有拿到有效值」。

這比單純 overfit、underfit、或 dataset 分布 shift 更嚴重。

## 第三輪哪些部分其實是成功的

第三輪並不是所有東西都失敗。

這一輪至少有兩個工程目標是成功的：

### 1. 可重現性改善

第三輪在同 seed、同配置、同 fold 下，search 與 final rerun 已經對齊，這代表第二輪最嚴重的「搜尋結果無法重現」問題明顯改善。

### 2. MPS 路徑沒有明顯亂跳

第三輪的 device probe 顯示：

- `device=auto` 最後解析到 `mps`
- probe 判定穩定
- 沒有因為不穩而退回 CPU

這代表第三輪不是因為裝置隨機性失控而導致結果差。

## 真正可能導致第三輪失敗的原因

以下是目前最合理的幾個候選原因，依優先順序排列。

### 1. context 路徑接入後的數值尺度失衡

第三輪新加入了 prefix-derived sequence context，並把 encoder 輸出與 context projection 串接後再送入 head。

這可能帶來兩種問題：

- context 特徵的尺度與 window encoder 輸出不匹配
- context projection 後的值過大或過小，讓 head 在一開始就進入不穩定區域

若這個問題存在，最直接的結果就是：

- validation loss 很早就變成 `NaN` 或失去可比較性
- best metric 永遠停在 `inf`

### 2. 新的 weighted loss 在某些 batch 上產生異常值

第三輪保留了 sequence-balanced 與 stage-aware weighting。如果權重和 context 路徑同時作用，可能導致：

- 某些 batch 的實際 loss 被放大
- backward 過程數值不穩
- 進而讓驗證過程從一開始就失效

尤其當 sample weight、standardization、context concat 三者同時變動時，最容易出現這類聯合作用。

### 3. early stopping / checkpoint 更新條件被新流程破壞

從 `best_epoch = 0`、`best_val_rmse = inf` 來看，也不能排除是程式流程本身有 bug，例如：

- validation 指標沒有被正確計算
- validation 指標有算出來，但沒有成功寫回 best tracker
- model checkpoint 條件在新 refactor 後沒有被正確觸發

這一類 bug 不一定會讓 train loss 爆炸，但會讓最終輸出看起來像模型完全沒學到。

### 4. 這次失敗不是 dataset 結構主導

`A2` 本來就有 train/test feed 不對稱和低 feed 長壽命稀有子群，`A3` 也確實有族內異質性，這些在前兩輪都成立。

但第三輪連 `A4` 這種最穩定的家族都一起崩掉，說明：

- dataset 結構仍然是背景難點
- 但不是這次全面失敗的主因

換句話說，第三輪不能解讀成「A3 還是太難」，而應解讀成「模型根本沒進入正常學習狀態」。

## 這次失敗對後續工作的意義

第三輪提供了兩個清楚的結論：

### 1. 第一輪基線仍應保留

第一輪正式基線目前仍是最可信的 benchmark 版本，不能被第三輪取代。

### 2. 下一步不該繼續疊功能，而應先除錯

在沒有先修掉第三輪訓練回歸之前，不適合繼續做以下事情：

- 進一步強化 `A3` 結構建模
- 新增更多 context 特徵
- 進一步擴大 family-specific tuning
- 把這條 benchmark 路徑接回 simulator 或 RL 主流程

## 建議的下一步除錯順序

若未來要重新開這條線，建議優先按以下順序處理：

1. 檢查 context standardization 後是否出現 `NaN`、`inf` 或極端值。
2. 檢查 model forward 的輸出尺度，確認 concat context 後沒有爆掉。
3. 檢查 weighted loss 的 batch 統計，確認沒有極端權重把 loss 推爆。
4. 檢查 validation metric 與 best-checkpoint 更新條件，確認不是流程 bug。
5. 只有在訓練重新恢復正常後，才回頭驗證 `A3` 的結構改良是否真的有效。

## 最後結論

第三輪效果差，最合理的總結是：

- 不是 benchmark 方向錯了
- 也不是單純 dataset 太難
- 而是新一輪為了改善可重現性與加入 sequence context 所做的重構，讓訓練流程本身出現了回歸

因此第三輪應被保留為一次失敗案例的記錄，但程式版本本身應退回到上一個已推播、可正常工作的版本，再另開除錯支線處理。
