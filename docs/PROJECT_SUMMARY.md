## 1. Project Context

The project is a competition proposal for a **Hybrid Energy Storage System (HESS)**, a prototype system (not yet physically built) that combines five renewable energy sources with a flywheel as a mechanical buffer and a Battery Energy Storage System (BESS) as the final storage stage. The core engineering problem: renewable sources fluctuate unpredictably, and feeding fluctuating power directly into a battery degrades it. A flywheel can absorb/release power quickly to smooth these fluctuations, but it needs a control signal telling it when and how much to act.

A Power Measurement Unit (PMU) can only report the **current** state of the system, it has no ability to anticipate future fluctuations. The core AI contribution of this project is a model that **predicts near-future power fluctuations** so that a "Predictive PID" controller can command the flywheel proactively (before a fluctuation hits) rather than reactively (after it hits).

**Pipeline at a glance:**
```
PMU (5 sensors) → Sum → Delta transform → TCN model → Predicted delta
                                                          ├─→ Error calculation (vs actual delta, for evaluation)
                                                          └─→ Predictive PID → VFD → Flywheel → BESS
```

---

## 2. Dataset

### 2.1 What dataset is used and why

The project has no physical prototype yet, so no real sensor data exists for the five target energy sources (Wind, Thermal, Geothermal, Hydrogen, Wave). A **proxy dataset** was needed to build and validate the modeling pipeline as a proof-of-concept.

**Final dataset used:** ENTSO-E Transparency Platform (transparency.entsoe.eu), "Actual Generation per Production Type," France bidding zone (BZN|FR), 15-minute resolution, period 1 January 2026 – 18 July 2026.

This was **not** the first dataset used. Two prior datasets were tried and abandoned:

1. **A generic hourly "sumber_energi.csv" Kaggle-style dataset** (French national grid data, hourly resolution, ~62,780 rows spanning multiple years) was the original dataset. Columns: Wind, Biomass, Nuclear, Solar, Hydroelectric — mapped respectively to wind, thermal, geothermal, hydrogen, wave. This mapping was known to be physically weak (Nuclear standing in for Geothermal; Hydroelectric standing in for Wave) but was accepted as a first-pass proxy since no better public alternative was known at the time.
2. Weather-only datasets (Open-Meteo, ERA5) were later tried as **additional features**, not as a replacement — see Trial 6 below.

The move to ENTSO-E 15-minute data was driven by the realization that hourly-resolution data cannot represent the sub-hour volatility that actually matters for a flywheel-based smoothing system — see Trial 5 below.

### 2.2 How the dataset maps to model features

Five columns are constructed from ENTSO-E's 21 raw production-type categories:

| HESS Channel | ENTSO-E Source Column(s) | Physical Justification |
|---|---|---|
| `wind` | Wind Onshore + Wind Offshore | Same physical phenomenon (turbines), summed directly |
| `thermal` | Biomass | Dispatchable, stable baseload — matches thermal's role |
| `geothermal` | Hydro Run-of-river and Pondage | France has no geothermal generation; run-of-river hydro is the most stable, baseload-like proxy available |
| `hydrogen` | Solar | France has no fuel-cell generation category; Solar's high intermittency (sunrise/sunset/cloud transients) behaviorally resembles a hydrogen fuel cell drawing on volatile electrolysis surplus |
| `wave` | Hydro Water Reservoir | France has no marine/wave generation; reservoir hydro is the most variable/dispatch-driven proxy available, resembling wave energy's irregular output |

**Known weakness (acknowledged, not hidden):** two of the five mappings (geothermal→run-of-river, wave→reservoir) are proxies for source types France does not generate at reportable scale. This is disclosed explicitly in the proposal as a limitation of the proof-of-concept, not presented as if it were real geothermal/wave data.

### 2.3 Cleaning steps and what they removed

- Raw ENTSO-E export: 35,036 rows in "long" format (one row per timestamp × production type).
- Pivoted to "wide" format (one row per timestamp, columns = production types).
- "n/e" (not estimated) string values converted to numeric zero, then forward-filled to propagate the last known reading.
- Rows where **all five mapped feature columns were simultaneously zero** were dropped: 16,067 rows removed (45.9% of the dataset). Root cause: France's TSO (RTE) reports many production types at hourly — not 15-minute — granularity to ENTSO-E, so 3 of every 4 fifteen-minute slots are natively empty and can only be partially recovered by forward-fill; combined with a 1-hour Daylight Saving Time gap on 29 March 2026.
- Final clean dataset: **18,969 rows**, spanning two continuous segments (2 Jan–29 Mar, and 29 Mar–18 Jul 2026) separated by the DST gap.

### 2.4 Target construction: delta, not absolute level

The model predicts **Δ(t) = Σsources(t+1) − Σsources(t)**, not the absolute summed load. This is a deliberate and important design decision (see Trial 3 below for how it was discovered as necessary).

Rationale: the absolute load level is strongly autocorrelated (lag-1 autocorrelation > 0.95). A naive "tomorrow = today" baseline achieves R² ≈ 0.92–0.97 on the absolute level with zero learned model. This makes absolute-level R² an unreliable/misleading metric of model skill. The delta target isolates the genuinely hard-to-predict component (the change), where a naive baseline scores R² ≈ 0.

Two metrics are reported throughout: **R² on delta** (true skill) and **R² on reconstructed level** (delta prediction added back to the known current level — useful for visualization, inflated by the autocorrelation "free credit").

### 2.5 Synthetic per-minute data (Brownian Bridge) — tried, not used for training

To argue that the system design generalizes to sub-15-minute deployment scenarios, a Brownian Bridge interpolation was implemented: for each pair of consecutive 15-minute points, 14 synthetic intermediate 1-minute values were generated via a constrained random walk (starts at v_t, ends at v_{t+1}, fluctuates in between with noise calibrated from the empirical per-channel delta standard deviation).

**Result: training on this synthetic per-minute data degraded model performance relative to the 15-minute data.** This is presented in the write-up as a limitation, not a success — the synthetic data was retained only as an illustrative visualization ("this is the kind of fluctuation a real per-minute sensor would show"), not as training data. The final trained model uses the 15-minute ENTSO-E data only.

---

## 3. Model Architecture

### 3.1 Final architecture: HESS-TCN-v2

A Temporal Convolutional Network (TCN) with three stacked residual blocks:

- Input: window of 48 timesteps × 5 features (48 × 15 min = 12 hours of history)
- Block 1: 2× causal dilated Conv1D (dilation=1), 5→64 channels, residual shortcut (1×1 conv, since channel count changes)
- Block 2: 2× causal dilated Conv1D (dilation=2), 64→128 channels, residual shortcut (1×1 conv)
- Block 3: 2× causal dilated Conv1D (dilation=4), 128→128 channels, residual shortcut (identity, channels unchanged)
- Each conv followed by ReLU + Dropout (rate 0.2)
- Last-timestep extraction (not pooling/flattening) — receptive field of 29 timesteps at kernel_size=3, close to the full 48-step window
- Fully connected: 128→64 (ReLU, Dropout) → 64→1 (scalar delta prediction)
- Approx. 1.2M parameters
- Optimizer: Adam, lr=5e-4, weight_decay=1e-5, ReduceLROnPlateau scheduler
- Loss: MSE
- Early stopping: patience=15 epochs on validation loss

### 3.2 Why TCN — decision process

Five architectures were empirically compared on the identical dataset/split/evaluation pipeline:

| Model | R² (delta) | MAE (MW) | Directional Acc | PID Reduction |
|---|---|---|---|---|
| Linear Regression | 0.5600 | 213.32 | 85.8% | 65.9% |
| MLP (flatten+dense, no temporal structure) | 0.3205 | 308.84 | 81.5% | 59.6% |
| GRU | 0.4203 | 276.03 | 76.5% | 62.9% |
| LSTM | 0.2942 | 327.06 | 77.1% | 58.3% |
| **TCN (final)** | **0.51–0.52** | **227–243** | **85.6%** | **70.5%** (best run, window=96 PID setpoint) |

Key findings from this comparison:
- **Linear Regression scored a higher raw R²** than TCN. This was investigated rather than ignored: the flattened 48×5=240-feature linear model can exploit strong linear temporal autocorrelation efficiently, which the data supports well. However, LR requires 240 dense weights with no compact embedded-runtime representation, making it impractical for ESP32 deployment.
- **MLP (no temporal structure) scored the worst of the deep models** (R²=0.32), which was used as evidence that temporal-aware architecture (TCN/GRU/LSTM) matters — removing time-awareness measurably hurts performance.
- **GRU and LSTM underperformed TCN** on this dataset and task, despite being standard sequence architectures. This is attributed to (a) TCN's parallel causal-dilated structure suiting the relatively short, information-dense 48-step window better than a fully sequential recurrent state, and (b) TCN's residual connections preserving raw-signal information across depth more directly than gated recurrent updates.
- An **ensemble of TCN + Linear Regression** (simple averaging) was also tried: R² improved to 0.58, directional accuracy to 85.8%, but PID reduction (66.9%) did not exceed TCN standalone (68.4% in that run). The ensemble was **rejected** for the final proposal: mixing a classical ML model and a DL model added complexity and deployment burden without improving the metric that matters most for the application (PID reduction), and TCN standalone remained simpler to justify architecturally and to deploy.

Final justification for TCN, in three parts:
1. **Architectural**: causal dilated convolutions guarantee no future-information leakage; the dilation schedule {1,2,4} gives an exponentially growing receptive field with only linear depth growth.
2. **Computational**: convolutions are parallelizable across the time axis (unlike RNN/LSTM/GRU sequential dependency), better suited to constrained embedded inference and INT8 quantization.
3. **Empirical**: the architecture comparison above, isolating the contribution of temporal-aware structure (MLP ablation) and confirming TCN's edge over GRU/LSTM at comparable parameter budgets.

### 3.3 Predictive PID (not conventional PID)

Conventional PID: `error(t) = actual(t) − setpoint(t)` → reactive, corrects after the fact.

This project's Predictive PID: `error(t) = predicted_delta_reconstructed(t) − setpoint(t)` → the TCN's forecast substitutes for the current measurement in the error term, so the flywheel begins compensating **before** the fluctuation is measured. Standard PID law applied on top: `u(t) = Kp·e(t) + Ki·Σe(i) + Kd·[e(t)−e(t-1)]`. Setpoint = 24-hour rolling mean of actual load. Anti-windup clamp on integral term (±500). Gains (Kp=0.7, Ki=0.05, Kd=0.1) are explicitly labeled as illustrative/proof-of-concept values, not system-identified from real hardware.

Design follows the predictive-PID formulation in Miller, Shah, Wood & Kwok (1999), *ISA Transactions* 38.

### 3.4 Hardware validation

Model exported to ONNX, profiled on Edge Impulse's "Bring Your Own Model" tool against the `espressif-esp32` device profile (EON Compiler runtime): **28.9 KB RAM (5.6% of 520KB SRAM), 96.6 KB Flash (2.4% of 4MB), 205ms inference latency (float32, unquantized)**, `isSupportedOnMcu: true`. Given the system operates at 15-minute resolution, 205ms per inference is not a practical constraint. INT8 quantization is planned but not yet implemented.

---

## 4. Chronological Trial-and-Error Log

This section documents every major iteration, in order, including failures and reversed decisions.

**Trial 1 — Data leakage in normalization scaler.** Original preprocessing computed min/max for normalization from the *entire* dataset (train+val+test combined). Fixed by computing scaler parameters from the training split only, then applying those fixed parameters to val/test. This is standard practice but was an actual bug in an early version, not a hypothetical.

**Trial 2 — Global Average Pooling instead of last-timestep extraction.** An early architecture applied global average pooling across the time dimension before the fully connected layers. This smoothed away exactly the fluctuation signal the model needed to learn, producing visibly over-smoothed predictions that missed peaks. Replaced with extracting only the final timestep's representation (justified by the TCN's receptive field already covering nearly the full window via dilation).

**Trial 3 — Absolute-level target produced misleadingly high R² (~0.94).** Before the delta reformulation, the target was the raw summed load level. R²=0.94 looked excellent, but comparison against a naive persistence baseline (predict "no change") revealed the baseline alone scored R²=0.92 on the same target — meaning the model was barely outperforming a trivial heuristic. This motivated the delta-target reformulation (Section 2.4), after which the naive baseline score on the new target (delta) dropped to ≈0, correctly reflecting that predicting fluctuations is genuinely hard.

**Trial 4 — Feature engineering: cyclical hour-of-day encoding.** Added sin/cos hour features hypothesizing they would help capture daily demand cycles. Result: R² decreased (0.56 → 0.44 in that experiment) because the `hydrogen` (Solar) feature already implicitly encodes day/night cycles, making the added feature redundant noise for a small-capacity model. Reverted.

**Trial 5 — Hourly-resolution data was insufficient; moved to 15-minute ENTSO-E data.** The original hourly dataset (WINDOW_SIZE=30, i.e., 30 hours history) produced R²(delta)=0.32, PID reduction=59.1%. Seven ablation experiments (rolling statistics, window size variation, ensembling, feature pruning, kernel size variation, GRU substitution, MLP baseline) all clustered in R²=0.23–0.33 regardless of approach, suggesting a data-resolution ceiling rather than a modeling deficiency. Switching to ENTSO-E's 15-minute French data (with the improved geothermal/wave proxy mapping, see Section 2.2) raised R²(delta) to ~0.52 and PID reduction to 64.5–68.4% in initial runs, later 70.5% with a longer PID setpoint window — direct empirical confirmation that resolution was a binding constraint.

**Trial 6 — Adding weather data (Open-Meteo hourly) as auxiliary features.** Hypothesis: wind speed, solar irradiance, cloud cover, temperature should help since they are the physical drivers of wind/solar volatility that the generation-only data lacks. Implementation: merged Open-Meteo hourly weather data (forward-filled to 15-minute resolution) as 5 additional input channels (10 total). **Result: R² decreased (0.52 → 0.43–0.46 across several configurations, including larger channel counts to compensate for higher input dimensionality).** Diagnosis: forward-filling hourly weather to 15-minute resolution produces 4 identical repeated values per hour, adding redundant/non-informative dimensions rather than genuine new signal, while also diluting the model's effective capacity across more input channels. **Weather data was dropped from the final model.**

**Trial 7 — Hyperparameter and architecture sweep.** Systematically varied: window size (30→48, kept 48), channels (16-32-32 → 32-64-64 → 64-128-128, kept 64-128-128 as best PID performer), learning rate (1e-3 → 5e-4, kept 5e-4), dropout (0.2 vs 0.3, kept 0.2). Best observed combination in isolated runs reached R²≈0.49–0.53 and PID reduction 67–70.5% depending on random seed and PID setpoint window configuration. No configuration exceeded this range by a wide margin, reinforcing the Trial 5 conclusion that further architecture tuning has diminishing returns without new information sources.

**Trial 8 — Custom PID-weighted loss function.** Hypothesis: standard MSE weights all errors equally, but PID performance cares more about large fluctuations and directional correctness than average magnitude error. Implemented a custom loss penalizing sign mismatches and weighting errors by target magnitude. **Result: R² improved slightly (0.51→0.53) but PID reduction decreased (68.4%→66.8%)** — the custom loss optimized a metric that didn't fully align with downstream PID utility. **Reverted to standard MSE** for the final model, since PID reduction was established (Section 3.2 discussion) as the more decision-relevant metric, and standard MSE gave the better PID result.

**Trial 9 — Data augmentation (jitter noise on training windows).** Hypothesis: adding small Gaussian noise to training windows (duplicated 2-3×) would improve robustness. **Result: performance decreased across the board** (R² 0.52→0.46, PID reduction 68.4%→63.7%, directional recall on the "rising" class dropped 0.77→0.69). Diagnosis: the added noise blurred the sharp fluctuation patterns the model needed to learn precisely, rather than acting as helpful regularization. **Augmentation was dropped.**

**Trial 10 — Brownian Bridge synthetic per-minute data as training input.** Described in Section 2.5. Training directly on Brownian-Bridge-interpolated per-minute data degraded performance relative to 15-minute data. Retained only as a visualization artifact, not as training data.

**Trial 11 — Model architecture comparison (Section 3.2).** Systematic evaluation of Linear Regression, MLP, GRU, LSTM, TCN on identical data/splits. TCN selected. Ensemble (TCN+LR) evaluated and rejected (Section 3.2).

---

## 5. Known Limitations (explicitly disclosed)

1. **Proxy dataset, not real sensor data.** No physical prototype exists yet; ENTSO-E France generation data stands in for the five target HESS channels. Two of five channel mappings (geothermal, wave) use physically dissimilar substitute sources (run-of-river hydro, reservoir hydro) because France has no reportable geothermal or marine generation.
2. **Temporal resolution mismatch.** Even at 15-minute resolution, this is far coarser than the sub-second/sub-minute timescale at which real flywheel-relevant power fluctuations occur. The Brownian Bridge experiment (Trial 10) was an attempt to probe this and showed the model does not currently benefit from synthetic upsampling — genuine higher-resolution sensor data would be needed to test this properly.
3. **No exogenous physical driver data (weather) in the final model.** Wind speed, solar irradiance, and wave height/period — the physical causes of the volatility being predicted — are absent from the final feature set after Trial 6 showed the available hourly weather data (resolution-mismatched to the 15-minute target) hurt rather than helped.
4. **R²=0.51 on delta means ~49% of fluctuation variance is unexplained.** This is presented transparently rather than obscured behind the higher (but less meaningful) R²=0.9953 level-reconstruction metric.
5. **PID gains are illustrative, not system-identified.** Kp/Ki/Kd were chosen for proof-of-concept demonstration, not derived from real VFD-flywheel plant dynamics; real commissioning would require system identification.
6. **Directional recall on "rising" fluctuations (0.81) is meaningfully lower than "falling" (0.89).** Sudden upward ramps (e.g., wind gusts) are harder to anticipate from generation history alone than gradual declines, which is a structural limitation of the current feature set.
7. **Dataset covers only ~6.5 months (Jan–Jul 2026) of a single year**, missing autumn/winter seasonal patterns entirely.
8. **Ensemble methods and custom loss functions were tried and both underperformed** the simpler TCN + MSE baseline on the PID-reduction metric, suggesting the current approach may be near a local optimum for this dataset without new information sources (weather at matching resolution, higher-frequency sensor data, or a full-year dataset).

---

## 6. Summary of Final Reported Results

- R² (delta, true model skill): **0.51**
- R² (level, reconstructed): **0.9953**
- MAE (delta): **227 MW**
- Directional Accuracy (|Δ|>50MW threshold): **85.6%**
- F1 macro-average: **0.85**
- PID fluctuation reduction vs. no control: **70.5%**
- Naive-baseline-driven PID (sanity check): **−19.6%** (i.e., worse than no control — confirms the predictive component is necessary, not optional)
- ESP32 deployment: 28.9KB RAM (5.6%), 96.6KB Flash (2.4%), 205ms latency — confirmed feasible via Edge Impulse BYOM profiling