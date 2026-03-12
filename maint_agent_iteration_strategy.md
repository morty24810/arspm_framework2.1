# Maint Agent 第一輪迭代策略

## 1. 現況審計

目前 repo 中的 maint stack 可以拆成三層來看。

### 1.1 Decision Policy Layer

- `DQN`
  - 由 [src/agents.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/src/agents.py) 中的 `MaintenanceAgentDDQN` 實作
  - 輸出動作為 `DN / IM / CM`
- `POMCP`
  - 由 [src/pomcp.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/src/pomcp.py) 中的 `POMCPPlanner` 實作
  - 以 generative rollout 的方式在有限 horizon 內做 lookahead

這兩者都是 `maintenance decision layer`，不是 `maintenance physics layer`。

### 1.2 Decision State Layer

目前的 maintenance state 由 [run_experiment.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/run_experiment.py) 中的 `build_maintenance_state()` 建立，核心特徵包括：

- `h`
- `dh`
- `eta`
- `slack_pressure`
- `local_urgency`
- `avg_slack`
- `lambda_hat`
- `ddt_hat`
- `risk_t`
- `win_e / win_l`
- `rul_mu / rul_sigma`
- `dt_last`
- `pf_dn / pf_im / pf_cm`

也就是說，`DQN` 與 `POMCP` 雖然是兩種不同決策器，但目前吃的是同一類 maintenance state。

### 1.3 Control Prior / Transition Layer

目前真正最強的人為先驗在於：

- `Hx / Hy`
  - 由 [config.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/config.py) 定義
  - 由 [run_experiment.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/run_experiment.py) 中的 `allowed_actions_by_region()` 與 `enforce_action_by_region()` 實際約束動作空間
- `IM` transition
  - 由 [src/env.py](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/src/env.py) 中的 `apply_maintenance()` 實作
  - 現況本質上仍是幾何式 baseline reset：每做一次 `IM`，下一個維修基線乘上 `0.8`
- replay-based health
  - 現況的健康度與 RUL 仍與固定 replay curve / 固定 lifespan 強耦合

因此，目前的 maint stack 雖然有 `DQN` 與 `POMCP` 兩個版本，但它們共享同一套控制先驗與 transition 假設。

### 1.4 對第一輪迭代的意義

第一輪真正需要改寫的，不是「把 `DQN` 換成 `POMCP`」或反過來，而是：

- state 來源
- 健康訊號的語義
- state 與新的 degradation model 的接口

## 2. 第一輪正式目標

第一輪 maint agent 迭代的正式目標不是推翻整個 maintenance policy，而是讓 decision layer 接上新的 `operation-conditioned degradation state`。

第一輪固定目標如下：

- 保留 `DQN` 與 `POMCP`
- 保留 `DN / IM / CM`
- 保留 `Hx/Hy on` 與 `Hx/Hy off` 作為正式對照軸
- 只替換 maint agent 所依賴的 `health / state definition`
- 不在第一輪宣稱新的 maintenance transition 已經具備物理真實性

第一輪的核心問題是：

- 當 degradation 已改成 `family-specific, operation-conditioned` 之後，maint agent 是否仍能在舊 state 下合理決策？

正式答案是：

- 不能直接沿用舊 state
- 但可以在保留既有動作空間與對照框架的前提下，先把 state 改寫成新的健康表示

## 3. 第一輪保留項

第一輪明確不動以下部分。

### 3.1 Action Space

- 維持 `DN / IM / CM`
- 不在第一輪新增 maintenance intensity、partial clean level 或其他擴充動作

### 3.2 Decision Policy Wrapper

- `DQN` 保留
- `POMCP` 保留
- 第一輪不做 `DQN` 與 `POMCP` 的勝負判定
- 第一輪把它們視為兩種 baseline-compatible decision wrapper

### 3.3 Reward / Cost 主架構

- 維持現有 time / material / risk / window violation 的成本骨架
- 不在第一輪重寫 maintenance reward 的 credit assignment
- 不在第一輪把 reward 直接改成物理損耗最小化公式

### 3.4 Online Decision Flow 主結構

- 維持 `maintenance decision before scheduling` 的大結構
- 維持 machine 在 decision point 上做 maintenance decision 的流程
- 維持 `region_on` 與 `region_off` 兩種實驗模式

## 4. 第一輪替換項

第一輪真正要替換的是 maint agent 所觀察到的健康訊號。

### 4.1 舊的核心來源

目前 maint state 的核心仍是：

- replay-based `h`
- 與 replay curve 綁定的 `RUL`
- 由固定 curve / fixed lifespan 派生出的風險 proxy

這和新的研究方向不一致，因為現在的主問題已改成：

- `machine 固定 family`
- `operation 提供 load`
- `family-specific degradation model`

### 4.2 新的核心來源

第一輪應把 maint state 的核心改為：

- `family-specific, operation-conditioned normalized health estimate`

這個 health estimate 必須來自新的 degradation model，並且能夠反映：

- `dust family`
- `dust_feed`
- `Flow_rate`
- `Differential_pressure`
- 壓差趨勢
- recent operation exposure history

### 4.3 RUL 的新角色

在新的 maint state 裡：

- `RUL` 仍可保留
- 但 `RUL` 應被視為由 latent health 推導出的衍生量
- `RUL` 不再是 state 的唯一核心
- state 的核心應轉為 `normalized health / degradation severity`

## 5. 第一輪 State 設計原則

第一輪的 state 設計應明確分成三類。

### 5.1 保留的舊資訊

以下資訊在第一輪仍然有價值：

- 維修歷史
- 距離上次維修的時間
- local urgency
- slack / queue 壓力相關特徵
- 現有 risk proxy

### 5.2 新增或替換的資訊

以下資訊應成為新的 maint state 核心：

- current normalized health
- health trend
- recent operation-family exposure
- recent `dust_feed` load
- family identity
- uncertainty

建議第一輪至少把以下量納入考慮：

- `current health`
- `delta health`
- `family id`
- `recent average dust_feed`
- `recent cumulative exposure`
- `Flow_rate`
- `Differential_pressure`
- `d(Differential_pressure)/dt`
- `state uncertainty`

### 5.3 應該刪弱的舊資訊

第一輪不必把 `Hx/Hy` 從系統中拿掉，但應降低以下資訊在 state 定義中的中心地位：

- 對 `Hx / Hy` 差值的過度依賴
- 對 raw replay lifespan 的過度依賴
- 對單一 black-box `RUL` 預測值的過度依賴

## 6. DQN 與 POMCP 的迭代差異

### 6.1 DQN

第一輪對 `DQN` 的主要影響是：

- 改 state definition
- 保留 action-value 形式
- 保留 `DN / IM / CM` 的離散動作空間

也就是說，第一輪不必重寫 `DQN` 的學習機制，主要是更換它看到的輸入語義。

### 6.2 POMCP

第一輪對 `POMCP` 的主要影響是：

- 先讓它吃新的 health state
- 暫時不重寫 maintenance transition physics
- generative model 的真正物理化改寫放到第二輪以後

這代表第一輪的 `POMCP` 仍然主要是：

- decision wrapper 保留
- 觀測與 belief 所依賴的健康訊號更新
- transition physics 暫時沿用舊骨架作過渡

### 6.3 第一輪的正式比較立場

第一輪不做以下事情：

- 不宣稱 `DQN` 比 `POMCP` 更適合最終物理模型
- 不用第一輪結果決定淘汰哪一套 policy

第一輪的正式立場是：

- `DQN` 與 `POMCP` 都先保留
- 先比較新的 state 進來之後，它們對硬先驗的依賴程度是否下降

## 7. `Hx/Hy on/off` 對照設計

第一輪維持 `Hx/Hy` 為正式對照軸。

### 7.1 為什麼不能直接拿掉

目前 `Hx/Hy` 雖然是人為先驗，但它同時也是：

- 現有 maint stack 的重要 baseline
- 你目前已有對照實驗的核心軸
- 判斷新 state 是否真的讓 agent 更少依賴硬門檻的重要比較基準

因此第一輪不應直接刪除 `Hx/Hy`。

### 7.2 第一輪正式實驗矩陣

第一輪固定保留以下四組：

- `DQN_on`
- `DQN_off`
- `POMCP_on`
- `POMCP_off`

其中：

- `on` 代表 `Hx/Hy` enforced
- `off` 代表 `Hx/Hy` 僅作參考，不作硬限制

### 7.3 第一輪比較重點

第一輪要看的不是哪一組數值最好，而是：

- 新的 health state 進來後，decision quality 對硬門檻的依賴是否下降
- `region_off` 下是否仍能做出有結構的 maintenance decision
- `DQN` 與 `POMCP` 是否都能從新的 state 中獲得穩定訊號

## 8. 第二輪以後才處理的內容

以下內容不進入第一輪。

### 8.1 `Hx/Hy` 軟邊界化

第二輪之後再考慮把：

- 硬式 `Hx / Hy`

改成：

- risk boundary
- cost boundary
- posterior-state-driven soft boundary

### 8.2 `IM` Transition Operator

第二輪之後再處理：

- 把 `IM = 0.8 * previous baseline` 改成 latent-state transition operator
- 讓 `IM` 效果變成可校準的 state reduction，而不是固定比例

### 8.3 更物理的 Reward 重寫

第二輪之後再考慮：

- 把 reward 明確連到 physical load
- 把 reward 明確連到 predicted post-maintenance gain
- 把 reward 明確連到 failure risk 與壓差成長

## 9. 驗收清單

- [ ] 文檔明確寫出 maint stack 的三層：`decision policy / decision state / control prior`
- [ ] 文檔明確寫出 `DQN` 與 `POMCP` 都是 decision layer，不是 maintenance physics layer
- [ ] 文檔明確寫出目前兩者共享同一類 maintenance state
- [ ] 文檔明確寫出目前 `Hx/Hy`、`IM=0.8` 與 replay-based health 都屬於控制先驗
- [ ] 文檔明確寫出第一輪要改的是 state / health definition
- [ ] 文檔明確寫出第一輪不改 `DN / IM / CM` action semantics
- [ ] 文檔明確寫出第一輪不改 reward 主體與 decision flow 主體
- [ ] 文檔明確寫出新 state 來自 `family-specific, operation-conditioned` degradation model
- [ ] 文檔明確寫出 `RUL` 在第一輪是衍生量，不是唯一核心 state
- [ ] 文檔明確寫出 `DQN_on / DQN_off / POMCP_on / POMCP_off` 四組對照
- [ ] 文檔明確寫出第二輪才處理 `Hx/Hy` 軟邊界化
- [ ] 文檔明確寫出第二輪才處理 `IM` transition operator
- [ ] 文檔明確寫出第二輪才處理更物理的 reward 重寫

## 10. 預設假設

- 文件語言使用繁體中文
- 第一輪 maint agent 迭代的目的，是讓 decision layer 接上新的 degradation state，而不是立刻推翻整個 maintenance policy
- `DQN` 與 `POMCP` 在第一輪地位相同，都是 baseline-compatible 的 decision wrapper
- 第一輪保留 `Hx/Hy on` 與 `Hx/Hy off`
- 第一輪不試圖宣稱新的 maintenance transition 已具物理真實性
- maintenance physics 的真正改寫留到第二輪與後續外部資料校準
