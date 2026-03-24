# Latest Result Analysis (2026-03-23)

## Scope

This note analyzes the latest paired experiment output:

- `outputs/paired_20260323_171437`

and compares it against two earlier result baselines already summarized in repo docs:

- `POMCP_DQN_result_root_cause_analysis.md`
  - based on `outputs/paired_20260310_083048`
- `BREAKDOWN_RESULT_ANALYSIS_AND_TOGGLE_NOTE.md`
  - based on `outputs/paired_20260311_103106`

## Important Comparison Caveat

The latest run is **not** a strict apples-to-apples continuation of the earlier runs.

Two major upstream changes were made before `paired_20260323_171437`:

1. RUL predictor was changed to a paper-aligned baseline:
   - only `Differential_pressure`
   - `GRU`
   - trained on machine 5 (`Data_No=18`) with `80/20` split
2. Machine mapping was changed to the first six machines from the paper-aligned subset:
   - `Data_No = 4, 8, 11, 17, 18, 23`

So the latest run should be interpreted as:

- a **new experimental stage**
- not merely a parameter retune of the old stage

## Latest Results Snapshot

Source:

- `outputs/paired_20260323_171437/paired_eval_rows.json`
- `outputs/paired_20260323_171437/paired_compare_rows.json`

### Full-system

| Route | Policy | Total | Tard | Maint | Breakdown | DN / IM / CM |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `region_on` | `DQN` | `2964.99` | `340.99` | `2624.00` | `2` | `275 / 4 / 0` |
| `region_on` | `POMCP` | `6960.79` | `411.06` | `6549.73` | `5` | `276 / 2 / 0` |
| `region_off` | `DQN` | `2594.78` | `222.78` | `2372.00` | `1` | `251 / 18 / 9` |
| `region_off` | `POMCP` | `3696.74` | `174.74` | `3522.00` | `1` | `188 / 73 / 15` |

### Maint-only

| Route | Policy | Total | Tard | Maint | Breakdown | DN / IM / CM |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `region_on` | `DQN` | `2735.53` | `111.53` | `2624.00` | `2` | `275 / 4 / 0` |
| `region_on` | `POMCP` | `8571.10` | `807.61` | `7763.50` | `6` | `276 / 1 / 0` |
| `region_off` | `DQN` | `1348.43` | `158.43` | `1190.00` | `0` | `246 / 19 / 10` |
| `region_off` | `POMCP` | `9874.85` | `567.94` | `9306.91` | `6` | `193 / 73 / 8` |

## Direct Interpretation Of Latest Run

### 1. DQN is still the safer winner overall

In the latest run, `DQN` beats `POMCP` on **all four main comparisons**:

- `region_on full-system`: `2964.99` vs `6960.79`
- `region_off full-system`: `2594.78` vs `3696.74`
- `region_on maint-only`: `2735.53` vs `8571.10`
- `region_off maint-only`: `1348.43` vs `9874.85`

This means the latest paper-style RUL change did **not** reverse the overall winner.

### 2. Constrained route is still bad for POMCP

Under `region_on`, `POMCP` is much worse:

- more breakdowns
- much higher maintenance/failure cost
- worse total cost

The constrained full-system delta is:

- `delta_total = +3995.80`
- `delta_breakdown = +3`

The constrained maint-only delta is even more severe:

- `delta_total = +5835.58`
- `delta_breakdown = +4`

So in the latest setup, `POMCP` is still not handling the constrained regime well.

### 3. Unrestricted full-system POMCP partially recovered, but still loses

This is the one place where the latest run shows a meaningful improvement in POMCP behavior.

Latest unrestricted full-system:

- `DQN`: `2594.78`
- `POMCP`: `3696.74`

Compared with the earlier `20260311` note:

- `DQN`: `2644.41`
- `POMCP`: `8065.33`

Interpretation:

- `POMCP` no longer collapses as badly as before in unrestricted full-system
- it even has slightly lower tardiness than `DQN`
- but it still spends too much on maintenance, so total cost remains worse

This suggests the latest RUL setup helped `POMCP` become less chaotic, but not yet cost-efficient.

### 4. Maint-only still says the main problem is maintenance policy, not scheduler

This remains the clearest signal in the repo.

In latest maint-only:

- `region_on`: `DQN 2735.53` vs `POMCP 8571.10`
- `region_off`: `DQN 1348.43` vs `POMCP 9874.85`

Because scheduler is fixed in maint-only compare, the gap is still mostly attributable to the maintenance layer itself.

## Comparison With Earlier Results

## A. Compared with `20260310` baseline

Source doc:

- `POMCP_DQN_result_root_cause_analysis.md`

That older run had the opposite unrestricted story:

| Route | DQN | POMCP | Earlier conclusion |
| --- | --- | --- | --- |
| `region_on full-system` | `774.41` | `995.82` | DQN slightly better |
| `region_off full-system` | `24884.41` | `431.62` | DQN collapsed, POMCP dominant |

Compared with latest:

- unrestricted `DQN` is no longer catastrophically collapsed
- unrestricted `POMCP` is no longer overwhelmingly dominant
- latest result is much more balanced, but still favors `DQN`

This is a real directional change.

Most likely interpretation:

- the old unrestricted comparison was heavily distorted by the earlier training/evaluation mismatch and policy collapse
- the newer setup makes the comparison more stable

## B. Compared with `20260311` breakdown-focused baseline

Source doc:

- `BREAKDOWN_RESULT_ANALYSIS_AND_TOGGLE_NOTE.md`

### Constrained full-system

| Version | DQN | POMCP |
| --- | ---: | ---: |
| `20260311` | `6066.05` | `9640.69` |
| `20260323` | `2964.99` | `6960.79` |

Change:

- both methods improved
- `DQN` improved much more
- `POMCP` is still clearly behind

### Unrestricted full-system

| Version | DQN | POMCP |
| --- | ---: | ---: |
| `20260311` | `2644.41` | `8065.33` |
| `20260323` | `2594.78` | `3696.74` |

Change:

- `DQN` is roughly stable but slightly better
- `POMCP` improved a lot
- but `POMCP` still loses on total cost

### Maint-only constrained

| Version | DQN | POMCP |
| --- | ---: | ---: |
| `20260311` | `1765.39` | `9922.07` |
| `20260323` | `2735.53` | `8571.10` |

Change:

- `POMCP` improved somewhat
- `DQN` worsened
- but the ranking did not change

### Maint-only unrestricted

| Version | DQN | POMCP |
| --- | ---: | ---: |
| `20260311` | `1679.92` | `12090.40` |
| `20260323` | `1348.43` | `9874.85` |

Change:

- both improved
- `DQN` improved further
- `POMCP` is still dramatically worse

## What The Latest Run Most Likely Means

### 1. The paper-aligned RUL predictor changed the scale, but not the winner

The latest paper-style RUL setup did not make `POMCP` beat `DQN`.

Instead, it changed the shape of the gap:

- constrained: still bad for `POMCP`
- unrestricted full-system: `POMCP` became less disastrous
- maint-only: `POMCP` still fails decisively

### 2. POMCP seems to trade tardiness for too much maintenance

Latest unrestricted full-system:

- `POMCP tard = 174.74`, better than `DQN 222.78`
- but `POMCP maint = 3522.00`, much worse than `DQN 2372.00`

So `POMCP` is not purely broken there.
It appears to be optimizing toward lower delay at the cost of much heavier intervention.

That is a more interesting failure mode than before.

### 3. The constrained route may now be too restrictive for the paper-style RUL estimator

Under `region_on`, both methods mostly stay near `DN`, with very few `IM/CM`, but `POMCP` still accumulates more breakdowns.

That suggests one of two things:

- the `Hx/Hy` gating still limits useful early action too much
- or the planner's internal timing/risk model is still misaligned with the real environment

Probably both are contributing.

## Recommendations

### Priority 1: treat `region_off + maint_only` as the cleanest decision benchmark

Right now this is the most interpretable comparison point:

- same scheduler
- unrestricted maintenance action space
- latest result still strongly favors `DQN`

If you need one headline result for the current stage, use this one.

### Priority 2: increase seeds before making a stronger claim

Current latest run still uses:

- `EXPERIMENT_SEEDS = (42,)`

So all aggregate stats have `std = 0`.
You should not write a strong paper conclusion from one seed.

Recommended next step:

- run at least `3-5` seeds
- compare whether latest unrestricted full-system `POMCP` improvement is stable

### Priority 3: keep constrained and unrestricted as separate research questions

Do not compress them into one winner/loser statement.

The latest evidence suggests:

- `region_on`: planner quality and gating interaction are still problematic
- `region_off`: planner is more reasonable, but still too expensive

These are not the same failure mode.

### Priority 4: investigate why POMCP over-maintains under the latest RUL setup

The latest run is no longer showing the old unrestricted collapse pattern.
Now the more useful question is:

- why does `POMCP` pay so much more maintenance cost even when breakdown count is equal or only slightly worse?

Concrete checks:

1. compare `IM` timing histograms between `DQN` and `POMCP`
2. compare decision states near `Hx/Hy`
3. inspect whether `POMCP` repeatedly repairs machines that would have survived with `DN`
4. inspect whether `expected_breakdown_loss` inside rollout is still overestimating intervention value

### Priority 5: write the paper claim conservatively

For the current codebase, the defensible statement is:

- under the latest paper-aligned RUL baseline and selected six-machine setup, `DQN` remains more cost-effective than `POMCP`
- `POMCP` shows partial recovery in unrestricted full-system tardiness, but still incurs excessive maintenance cost
- maint-only comparisons indicate the main residual weakness lies in the maintenance decision layer rather than the scheduler

## Bottom Line

If the question is "what does the latest result say?", the shortest honest answer is:

- the latest paper-aligned RUL change improved result stability
- it especially softened the old unrestricted `POMCP` failure pattern
- but it did **not** overturn the overall ranking
- `DQN` is still the stronger maintenance policy in the current implementation
- the cleanest supporting evidence is still the maint-only comparison
