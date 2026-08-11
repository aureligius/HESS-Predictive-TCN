# HESS Predictive TCN + PID

AI-driven predictive control system for a Hybrid Energy Storage System (HESS) prototype, combining a Temporal Convolutional Network (TCN) for short-horizon renewable power forecasting with a Predictive PID controller for flywheel-based fluctuation smoothing.

Built for **DELTACUP**.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Dataset](#dataset)
  - [Delta-Target Formulation](#delta-target-formulation)
  - [Model Architecture](#model-architecture)
  - [Predictive PID](#predictive-pid)
- [Model Comparison](#model-comparison)
- [Hardware Validation](#hardware-validation)
- [Trial and Error Log](#trial-and-error-log)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Overview

A Hybrid Energy Storage System (HESS) combines five renewable energy sources (Wind, Thermal, Geothermal, Hydrogen, Wave) with a flywheel as a mechanical buffer and a Battery Energy Storage System (BESS) as final storage. Renewable sources fluctuate unpredictably; feeding raw fluctuating power directly into a battery accelerates degradation.

A Power Measurement Unit (PMU) can only report the **current** state of the system, it cannot anticipate what happens next. This project adds an AI forecasting layer that predicts near-future power fluctuations, allowing a **Predictive PID controller** to command the flywheel proactively (before a fluctuation is measured) instead of reactively (after it hits).

```
PMU (5 sensors) → Sum → Delta Transform → TCN Model → Predicted Delta
                                                          ├─→ Error Calculation (vs actual delta, evaluation only)
                                                          └─→ Predictive PID → VFD → Flywheel → BESS
```

No physical prototype exists yet. This repository implements and validates the AI + control pipeline as a proof-of-concept using a public proxy dataset, with results intended to inform physical hardware development.

---

## System Architecture

| Stage | Component | Role |
|---|---|---|
| Input | 5 sensor channels (Wind, Thermal, Geothermal, Hydrogen, Wave) | Raw generation readings |
| Preprocessing | Sum → Delta transform → Normalize → Windowing | Converts raw readings into model-ready sequences |
| Forecasting | HESS-TCN-v2 (Temporal Convolutional Network) | Predicts next-interval power delta |
| Control | Predictive PID | Converts forecast into corrective signal |
| Actuation | VFD → Flywheel motor | Physically smooths power delivered to BESS |

---

## Results Summary

| Metric | Value |
|---|---|
| R² (delta — true model skill) | **0.51** |
| R² (level, reconstructed) | 0.9953 |
| MAE (delta) | 227 MW |
| Directional Accuracy (\|Δ\|>50MW) | **85.6%** |
| F1 macro-average | 0.85 |
| PID fluctuation reduction vs. no control | **70.5%** |
| Naive-baseline-driven PID (sanity check) | −19.6% (confirms AI forecasting is necessary, not optional) |
| ESP32 RAM usage | 28.9 KB (5.6% of 520KB SRAM) |
| ESP32 Flash usage | 96.6 KB (2.4% of 4MB) |
| Inference latency | 205 ms (unquantized) |

See [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) for full methodology, the complete trial-and-error log, and known limitations.

---

## Repository Structure

```
DELTACUP/
│
├── data/
│   └── sumber_energi_15.csv        # Raw ENTSO-E generation data (15-min resolution)
│
├── src/
│   ├── config.py                   # Global config (USE_SYNTHETIC flag)
│   ├── model.py                    # HESS-TCN-v2 architecture (single source of truth)
│   ├── explore.py                  # Preprocessing: raw CSV → windowed .npy tensors
│   ├── train.py                    # Model training with early stopping
│   ├── show.py                     # Model evaluation, reconstruction, plotting
│   ├── pid.py                      # Predictive PID simulation
│   └── confusion_matrix.py         # Directional accuracy + confusion matrix
│
├── outputs/                        # Generated at runtime (not committed)
│   ├── *.npy                       # Preprocessed tensors, scaler params
│   ├── best_model.pth              # Trained model checkpoint
│   ├── loss_history.npy            # Train/val loss per epoch
│   └── figures/
│       ├── fig_pred_vs_actual.png
│       ├── fig_scatter_delta.png
│       ├── fig_pid_simulation.png
│       ├── confusion_matrix.png
│       └── fig_loss_curve.png
│
├── docs/
│   └── PROJECT_SUMMARY.md          # Full methodology + trial-and-error log for review
│  
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Create a virtual environment (Python 3.11 required)

```bash
py -3.11 -m venv hess_env
hess_env\Scripts\activate        # Windows
# source hess_env/bin/activate   # macOS/Linux
```

### 2. Install PyTorch

**Standard GPUs (RTX 40-series and earlier, or CPU-only):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**RTX 50-series / Blackwell GPUs (sm_120)** — requires PyTorch nightly, as stable builds do not yet support this architecture:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify GPU detection:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run scripts in order from the `src/` directory. Each stage depends on outputs from the previous one.

```bash
cd src

python explore.py            # 1. Preprocess raw data → .npy tensors
python train.py              # 2. Train HESS-TCN-v2 → best_model.pth
python show.py                # 3. Evaluate on test set → R², MAE, plots
python pid.py                  # 4. Simulate Predictive PID → fluctuation reduction %
python confusion_matrix.py      # 5. Directional accuracy + confusion matrix
```

`show.py` is also imported directly by `pid.py` (via `build_results()`), so running `pid.py` alone will re-run evaluation automatically if needed.

### Configuration

Edit `src/config.py` to toggle between real 15-minute data and synthetic per-minute (Brownian Bridge) data:

```python
USE_SYNTHETIC = False   # False = 15-minute ENTSO-E data (recommended, best results)
                        # True  = synthetic per-minute data (visualization/experimental only)
```

**Note:** training on `USE_SYNTHETIC=True` data produces degraded model performance (see [Trial and Error Log](#trial-and-error-log)). It is retained for visualization purposes only, demonstrating the type of sub-15-minute fluctuation a physical sensor deployment would need to handle.

---

## Methodology

### Dataset

Source: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu), "Actual Generation per Production Type," France bidding zone (BZN\|FR), 15-minute resolution, 1 January – 18 July 2026.

**Why France / ENTSO-E:** No physical prototype exists yet, so a public proxy dataset was required. ENTSO-E was selected over an earlier hourly-resolution dataset because sub-hourly volatility is essential for a flywheel-smoothing use case (see Trial 5 below).

**Cleaning:** 35,036 raw rows → 18,969 rows after removing intervals where all five mapped source channels report zero (45.9% of rows), caused primarily by RTE's hourly (not 15-minute) reporting granularity for several source categories to ENTSO-E, plus a 1-hour Daylight Saving Time gap on 29 March 2026.

**Feature mapping** (ENTSO-E category → HESS channel):

| HESS Channel | ENTSO-E Source | Rationale |
|---|---|---|
| `wind` | Wind Onshore + Wind Offshore | Same physical phenomenon, summed |
| `thermal` | Biomass | Stable, dispatchable baseload |
| `geothermal` | Hydro Run-of-river and Pondage | France has no geothermal; run-of-river is the most stable available proxy |
| `hydrogen` | Solar | France has no fuel-cell generation; solar's high intermittency behaviorally resembles a hydrogen fuel cell drawing on volatile surplus |
| `wave` | Hydro Water Reservoir | France has no marine generation; reservoir hydro's dispatch-driven variability is the closest available proxy |

This mapping is an explicit, disclosed limitation of the proof-of-concept — see [Known Limitations](#known-limitations).

### Delta-Target Formulation

The model predicts **change** in aggregate load, not absolute level:

```
Δ(t) = Σ_sources(t+1) − Σ_sources(t)
```

**Why:** the absolute load level is strongly autocorrelated (lag-1 autocorrelation > 0.95). A trivial "tomorrow = today" baseline achieves R² ≈ 0.92–0.97 on the absolute level with zero learned model, making level-R² a misleading skill metric. Under the delta formulation, the same naive baseline scores R² ≈ 0, correctly reflecting that fluctuation prediction is genuinely difficult.

Two metrics are reported throughout: **R² on delta** (true predictive skill) and **R² on reconstructed level** (delta + known current level, useful for visualization but inflated by autocorrelation).

### Model Architecture

**HESS-TCN-v2** — three stacked causal-dilated residual TCN blocks:

```
Input (48 × 5)                          # 12 hours history, 5 features
  → Block 1: 2× CausalConv1D (dilation=1), 5→64 ch, residual (1×1 conv)
  → Block 2: 2× CausalConv1D (dilation=2), 64→128 ch, residual (1×1 conv)
  → Block 3: 2× CausalConv1D (dilation=4), 128→128 ch, residual (identity)
  → Last-timestep extraction               # receptive field ≈ 29 steps
  → FC: 128→64 (ReLU, Dropout) → 64→1      # scalar delta prediction
```

Each convolution is followed by ReLU and Dropout (0.2). Causality is enforced via left-padding with right-side trimming, guaranteeing no future-information leakage. The dilation schedule {1, 2, 4} produces an exponentially growing receptive field with only linear depth growth.

**Training:** Adam optimizer (lr=5×10⁻⁴, weight_decay=10⁻⁵), MSE loss, ReduceLROnPlateau scheduler, early stopping (patience=15 epochs), best-validation-loss checkpoint retained.

### Predictive PID

A conventional PID controller reacts to the **current measured** deviation:

```
e_conventional(t) = actual(t) − setpoint(t)          [reactive]
```

This implementation substitutes the TCN's forecast for the current measurement:

```
e_predictive(t) = predicted(t) − setpoint(t)         [anticipatory]

u(t) = Kp·e(t) + Ki·Σe(i) + Kd·[e(t) − e(t−1)]
```

Setpoint = 24-hour rolling mean of actual load. Integral term is anti-windup clamped (±500). Gains `Kp=0.7, Ki=0.05, Kd=0.1` are illustrative proof-of-concept values — physical commissioning would require system identification on the real VFD-flywheel plant.

Design follows the predictive PID formulation established in Miller, Shah, Wood & Kwok (1999), *ISA Transactions* 38.

---

## Model Comparison

Five architectures were evaluated on identical data, splits, and evaluation pipeline:

| Model | R² (delta) | MAE (MW) | Directional Acc | PID Reduction |
|---|---|---|---|---|
| Linear Regression | 0.56 | 213.32 | 85.8% | 65.9% |
| MLP (no temporal structure) | 0.32 | 308.84 | 81.5% | 59.6% |
| GRU | 0.42 | 276.03 | 76.5% | 62.9% |
| LSTM | 0.29 | 327.06 | 77.1% | 58.3% |
| **TCN (selected)** | **0.51–0.52** | **227–243** | **85.6%** | **70.5%** |

**Why TCN was selected despite Linear Regression's higher raw R²:**

1. **Architectural** — causal dilated convolutions guarantee zero future-information leakage; exponential receptive field growth with only linear parameter growth.
2. **Computational** — fully parallelizable across the time axis (unlike RNN/LSTM/GRU sequential dependency), well-suited to embedded inference and INT8 quantization. Linear Regression's 240-weight flat structure has no compact embedded-runtime representation.
3. **Empirical** — TCN achieves the best PID-reduction outcome, the metric most directly tied to system utility, despite a marginally lower delta-R² than Linear Regression.

An **ensemble of TCN + Linear Regression** (simple averaging) was also evaluated: R² improved to 0.58, but PID reduction (66.9%) did not exceed TCN standalone. The ensemble was rejected in favor of the simpler, single-model TCN pipeline.

---

## Hardware Validation

The trained model was exported to ONNX and profiled via Edge Impulse's "Bring Your Own Model" (BYOM) tool against the `espressif-esp32` device profile:

| Resource | Usage | % of ESP32 Capacity |
|---|---|---|
| RAM (tensor arena) | 28.9 KB | 5.6% of 520 KB SRAM |
| Flash (ROM) | 96.6 KB | 2.4% of 4 MB |
| Inference latency | 205 ms | — |

`isSupportedOnMcu: true`. Given the system operates at 15-minute resolution, 205ms latency is operationally inconsequential (≈900 seconds between prediction cycles). INT8 post-training quantization is planned for physical implementation but not yet applied.

---

## Trial and Error Log

A condensed summary — full details in [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md).

| # | Trial | Outcome |
|---|---|---|
| 1 | Global min/max scaler fit on entire dataset | **Bug** — caused data leakage; fixed to fit on train split only |
| 2 | Global Average Pooling before FC layers | **Failed** — smoothed away fluctuation signal; replaced with last-timestep extraction |
| 3 | Absolute-level target (R²=0.94) | **Misleading** — naive baseline scored R²=0.92 on same target; motivated delta reformulation |
| 4 | Cyclical hour-of-day features | **Failed** — R² 0.56→0.44; redundant with existing solar-derived day/night signal |
| 5 | Hourly → 15-minute resolution switch | **Success** — R² 0.32→0.52, PID reduction 59%→70% |
| 6 | Weather data (Open-Meteo, hourly) as features | **Failed** — R² 0.52→0.43–0.46; hourly-to-15min forward-fill added redundant, non-informative dimensions |
| 7 | Hyperparameter sweep (window, channels, LR, dropout) | Converged on window=48, channels=(64,128,128), lr=5e-4, dropout=0.2 |
| 8 | Custom PID-weighted loss function | **Failed** — R² improved but PID reduction decreased; reverted to standard MSE |
| 9 | Data augmentation (jitter noise) | **Failed** — degraded performance across all metrics; noise blurred sharp fluctuation patterns |
| 10 | Brownian Bridge synthetic per-minute training | **Failed** — degraded performance vs. 15-minute data; retained for visualization only |
| 11 | Architecture comparison (5 models) + ensemble | TCN selected; ensemble tried and rejected |

---

## Known Limitations

1. **Proxy dataset, not real sensor data** — no physical prototype exists; two of five channel mappings (geothermal, wave) substitute physically dissimilar sources due to unavailability in France's generation mix.
2. **15-minute resolution is coarser than real flywheel-relevant fluctuation timescales** (sub-second to sub-minute). The Brownian Bridge experiment attempted to probe this and showed no benefit from synthetic upsampling alone.
3. **No exogenous physical driver data (weather) in the final model** — the available hourly weather data, resolution-mismatched to the 15-minute target, degraded rather than improved performance.
4. **R²=0.51 on delta means ~49% of fluctuation variance is unexplained** — reported transparently alongside the higher but less meaningful level-reconstruction R².
5. **PID gains are illustrative**, not system-identified from real hardware.
6. **Directional recall on rising fluctuations (0.81) is lower than falling (0.89)** — sudden upward ramps are structurally harder to anticipate from generation history alone.
7. **Dataset covers only ~6.5 months** (Jan–Jul 2026), missing autumn/winter seasonal patterns.
8. **Ensemble methods and custom loss functions were tried and underperformed** the simpler TCN + MSE baseline, suggesting the current approach may be near a local optimum without new information sources.

---

## References

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271.
- Miller, R. M., Shah, S. L., Wood, R. K., & Kwok, E. K. (1999). *Predictive PID.* ISA Transactions, 38, 11–23.
- ENTSO-E Transparency Platform. [https://transparency.entsoe.eu](https://transparency.entsoe.eu)
- Edge Impulse. [https://edgeimpulse.com](https://edgeimpulse.com)

---

## License

*(Add your license here, e.g., MIT, or leave as competition-internal if not intended for public reuse.)*

## Acknowledgments

Built for the DELTACUP competition. See `docs/PROJECT_SUMMARY.md` for the complete technical writeup intended for external/AI review.
