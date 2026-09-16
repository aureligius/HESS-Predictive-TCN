# baseline_arima.py
"""
Baseline STATISTIK MURNI (tanpa AI/deep learning) untuk dibandingkan
dengan TCN.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from config import USE_SYNTHETIC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "output"
CSV_PATH = DATA_DIR / "sumber_energi_15.csv"

# STEP 1
df_raw = pd.read_csv(CSV_PATH)
df_raw["DateTime"] = df_raw["MTU (CET/CEST)"].str.split(" - ").str[0]
df_raw["DateTime"] = df_raw["DateTime"].str.replace(
    r"\s*\(CET\)|\s*\(CEST\)", "", regex=True).str.strip()
df_raw["DateTime"] = pd.to_datetime(df_raw["DateTime"], format="%d/%m/%Y %H:%M:%S")

df_wide = df_raw.pivot_table(
    index="DateTime", columns="Production Type",
    values="Generation (MW)", aggfunc="first",
).reset_index()
df_wide.columns.name = None
for col in df_wide.columns:
    if col != "DateTime":
        df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce").fillna(0)
df_wide = df_wide.ffill().fillna(0)
df_wide = df_wide.sort_values("DateTime").reset_index(drop=True)

df = pd.DataFrame()
df["DateTime"] = df_wide["DateTime"]
df["wind"] = df_wide["Wind Onshore"] + df_wide["Wind Offshore"]
df["thermal"] = df_wide["Biomass"]
df["geothermal"] = df_wide["Hydro Run-of-river and pondage"]
df["hydrogen"] = df_wide["Solar"]
df["wave"] = df_wide["Hydro Water Reservoir"]

FEATURE_COLS = ["wind", "thermal", "geothermal", "hydrogen", "wave"]
for col in FEATURE_COLS:
    df[col] = df[col].clip(lower=0)
all_zero = (df[FEATURE_COLS] == 0).all(axis=1)
df = df[~all_zero].copy().reset_index(drop=True)

df["load_output"] = df[FEATURE_COLS].sum(axis=1)
df["load_output_level"] = df["load_output"]
df["load_output"] = df["load_output"].diff().shift(-1)
df = df.iloc[:-1].reset_index(drop=True)

print(f"[OK] Preprocessing selesai. Shape: {df.shape}")

# ============================================================
# STEP 2 — Split 70/15/15 (SAMA seperti explore.py, TANPA windowing)
# ============================================================
n_rows = len(df)
train_end = int(n_rows * 0.70)
val_end = int(n_rows * 0.85)

delta_train = df["load_output"].iloc[:train_end].values
delta_test = df["load_output"].iloc[val_end:].values
level_test_last = df["load_output_level"].iloc[val_end:].values

print(f"[OK] Train: {len(delta_train)}, Test: {len(delta_test)}")

# ============================================================
# STEP 3 — Fit ARIMA pada train, forecast rolling 1-step di test
# ============================================================
# ARIMA(p,d,q) — d=0 karena data SUDAH dalam bentuk delta (sudah "differenced")
# Order (2,0,2) dipilih sebagai starting point ilustratif, bukan hasil grid search
ARIMA_ORDER = (2, 0, 2)

print(f"[INFO] Fitting ARIMA{ARIMA_ORDER} pada data training...")
model_fit = ARIMA(delta_train, order=ARIMA_ORDER).fit()
print("[OK] ARIMA fitted.")

# Rolling forecast: prediksi 1 langkah, lalu "kasih tahu" nilai aktual,
# ulangi — TANPA refit ulang parameter (refit=False, jauh lebih cepat)
print("[INFO] Menjalankan rolling one-step forecast pada test set...")
predictions = []
current_model = model_fit
for i in range(len(delta_test)):
    pred = current_model.forecast(steps=1)[0]
    predictions.append(pred)
    # Update model dengan nilai aktual test[i], tanpa refit parameter
    current_model = current_model.append([delta_test[i]], refit=False)

delta_pred_arima = np.array(predictions)
print("[OK] Forecasting selesai.")

# ============================================================
# STEP 4 — Evaluasi (metrik SAMA seperti show.py, supaya bisa dibandingkan)
# ============================================================
r2_delta = r2_score(delta_test, delta_pred_arima)
mae_delta = mean_absolute_error(delta_test, delta_pred_arima)
mae_naive = mean_absolute_error(delta_test[1:], delta_test[:-1])
mase_delta = mae_delta / mae_naive

level_actual_next = level_test_last + delta_test
level_pred_recon = level_test_last + delta_pred_arima
r2_level = r2_score(level_actual_next, level_pred_recon)

THRESHOLD = 50
mask = np.abs(delta_test) > THRESHOLD
actual_dir = (delta_test[mask] > 0).astype(int)
pred_dir = (delta_pred_arima[mask] > 0).astype(int)
dir_acc = (actual_dir == pred_dir).mean()

print("\n" + "=" * 55)
print(f"   HASIL EVALUASI ARIMA{ARIMA_ORDER} (BASELINE STATISTIK MURNI)")
print("=" * 55)
print(f"  R2 pada DELTA          : {r2_delta:.4f}")
print(f"  MAE pada DELTA         : {mae_delta:.2f} MW")
print(f"  MASE pada DELTA        : {mase_delta:.4f}")
print(f"  R2 pada LEVEL (recon)  : {r2_level:.4f}")
print(f"  Directional Accuracy   : {dir_acc:.4f}")
print("=" * 55)

# ============================================================
# STEP 5 — Simulasi PID (SAMA seperti pid.py, untuk baris tabel yang adil)
# ============================================================
WINDOW = 96
setpoint = np.convolve(level_actual_next, np.ones(WINDOW) / WINDOW, mode="same")
Kp, Ki, Kd = 0.7, 0.05, 0.1
INTEGRAL_CLAMP = 500
integral, prev_error = 0.0, 0.0
u = np.zeros(len(level_actual_next))
for t in range(len(level_actual_next)):
    error = level_pred_recon[t] - setpoint[t]
    integral = np.clip(integral + error, -INTEGRAL_CLAMP, INTEGRAL_CLAMP)
    derivative = error - prev_error
    u[t] = Kp * error + Ki * integral + Kd * derivative
    prev_error = error
smoothed = level_actual_next - u
dev_raw = np.mean(np.abs(level_actual_next - setpoint))
dev_smoothed = np.mean(np.abs(smoothed - setpoint))
reduction_pct = (1 - dev_smoothed / dev_raw) * 100

print(f"\n  PID Reduction (ARIMA-driven) : {reduction_pct:.1f}%")
print("=" * 55)
print("\n[INFO] Bandingkan angka di atas dengan Table 1 (TCN, LR, GRU, dst)")
print("[INFO] untuk baris baru 'ARIMA (statistical baseline)'")