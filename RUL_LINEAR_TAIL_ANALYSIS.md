# RUL Linear Tail Analysis

## Why the plateau appears

- The labeled `Test_Data_CSV.csv` sequences are right-censored, not run-to-failure.
- For every `Data_No`, `Time + RUL` is constant and `RUL` drops by exactly `0.1` per row.
- No `Test` sequence reaches terminal `RUL = 0`.
- The simulator currently replays observed sensor windows and clamps any index beyond the dataset tail to the last observed sample.
- That means the latent "true RUL" used by the simulator also plateaus once replay reaches the last sample, even though the labeled dataset implies a missing linear tail.

## Representative machines from the current six-machine mapping

Current machine mapping: `Data_No = 4, 8, 11, 17, 18, 23`.

Two representative examples:

| Simulator machine | `Data_No` | Observed rows | Labeled total life (`Time + RUL`) | Last labeled `RUL` | Last normalized `RUL` | Missing linear tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 4  | 608  | 62.8  | 1.9 | 0.030255 | 19 more dataset steps to zero |
| M5 | 18 | 2579 | 263.2 | 5.3 | 0.020137 | 53 more dataset steps to zero |

Interpretation:

- Under the old plateau logic, both machines would stay forever at their terminal observed normalized RUL once replay exhausted.
- Under the new tail logic, they keep the observed replay/GRU value up to the last sample, then decay linearly to `0`.
- This replaces only the missing censored tail. It does not fabricate new sensor trajectories.

## Intended simulator semantics

- Observed segment: unchanged replay/GRU behavior.
- Beyond the last observed sample: latent true RUL continues linearly downward.
- Observed/noisy RUL used for monitoring can still stay replay-based and clamped.
- This keeps the simulator aligned with the paper-style observed segment while avoiding an artificial non-decreasing tail.
