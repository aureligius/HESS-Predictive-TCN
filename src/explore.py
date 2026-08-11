from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import USE_SYNTHETIC    

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "sumber_energi_15.csv"
HORIZON = 1
if USE_SYNTHETIC: 
    WINDOW_SIZE = 240
else:
    WINDOW_SIZE = 48

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

# ============================================================
# STEP 1 — LOAD DAN RESHAPE (long → wide)
# ============================================================
df_raw = pd.read_csv(CSV_PATH)

df_raw["DateTime"] = df_raw["MTU (CET/CEST)"].str.split(" - ").str[0]
df_raw["DateTime"] = df_raw["DateTime"].str.replace(
    r"\s*\(CET\)|\s*\(CEST\)", "", regex=True).str.strip()
df_raw["DateTime"] = pd.to_datetime(df_raw["DateTime"], format="%d/%m/%Y %H:%M:%S")

df_wide = df_raw.pivot_table(
    index="DateTime",
    columns="Production Type",
    values="Generation (MW)",
    aggfunc="first",
).reset_index()
df_wide.columns.name = None

for col in df_wide.columns:
    if col != "DateTime":
        df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce").fillna(0)
df_wide = df_wide.ffill().fillna(0)
df_wide = df_wide.sort_values("DateTime").reset_index(drop=True)

# ============================================================
# STEP 2 — MAPPING KE 5 SUMBER HESS + BERSIHKAN ZERO
# ============================================================
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
print(f"[OK] Setelah buang zero: {len(df)} baris (per-15-menit)")

# ============================================================
# STEP 3 — SYNTHETIC PER-MINUTE (BROWNIAN BRIDGE)
# ============================================================
def generate_synthetic_per_minute(df_15min, feature_cols, sigma_scale=0.3, seed=42):
    np.random.seed(seed)
    STEPS = 15

    rows = []
    for i in range(len(df_15min) - 1):
        t_start = df_15min.iloc[i]
        t_end = df_15min.iloc[i + 1]
        row_minute = {}
        for col in feature_cols:
            v_start = t_start[col]
            v_end = t_end[col]
            delta = v_end - v_start
            col_std = df_15min[col].diff().dropna().std()
            sigma = col_std * sigma_scale / np.sqrt(STEPS)
            random_increments = np.random.normal(0, sigma, STEPS)
            adjustment = (delta - random_increments.sum()) / STEPS
            adjusted_increments = random_increments + adjustment
            values = [v_start]
            for inc in adjusted_increments[:-1]:
                values.append(max(0, values[-1] + inc))
            row_minute[col] = values
        rows.append(row_minute)

    minute_data = {col: [] for col in feature_cols}
    minute_data["DateTime"] = []
    start_time = pd.to_datetime(df_15min["DateTime"].iloc[0])

    for i, row in enumerate(rows):
        for m in range(STEPS):
            minute_data["DateTime"].append(start_time + pd.Timedelta(minutes=i * 15 + m))
            for col in feature_cols:
                minute_data[col].append(row[col][m])

    return pd.DataFrame(minute_data)

# df_clean = DataFrame SEBELUM delta transform — ini yang masuk ke Brownian Bridge
df_clean = df.copy()

if USE_SYNTHETIC:
    df_work = generate_synthetic_per_minute(df_clean, FEATURE_COLS, sigma_scale=0.3)
    print(f"[OK] Synthetic per-menit: {len(df_work)} baris")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_work["wind"][:120], label="Per-menit sintetis", linewidth=0.8, color="#534AB7")
    ax.scatter(range(0, 121, 15), df_clean["wind"].iloc[:9].values, color="red", zorder=5, s=40, label="Data asli (15-menit)")
    ax.set_title("Wind Power: Data Asli vs Sintetis Per-Menit (Brownian Bridge)")
    ax.set_xlabel("Menit")
    ax.set_ylabel("MW")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "synthetic_vs_real.png", dpi=150)
    plt.close()
    print("[OK] Plot tersimpan → synthetic_vs_real.png")
    resolution_label = "per-menit sintetis"
else:
    df_work = df_clean.copy()
    resolution_label = "per-15-menit asli"

print(f"[OK] Dataset yang dipakai untuk training: {resolution_label}")
print(f"     Shape: {df_work.shape}")

# ============================================================
# STEP 4 — TARGET DELTA
# ============================================================
TARGET_COL = "load_output"
ALL_COLS = FEATURE_COLS + [TARGET_COL]

df_work["load_output"] = df_work[FEATURE_COLS].sum(axis=1)
df_work["load_output_level"] = df_work["load_output"]
df_work["load_output"] = df_work["load_output"].diff().shift(-1)
df_work = df_work.iloc[:-1].reset_index(drop=True)
print(f"[OK] Delta target dibuat. Shape: {df_work.shape}")

# ============================================================
# STEP 5 — SPLIT 70/15/15
# ============================================================
n_rows = len(df_work)
train_end = int(n_rows * 0.70)
val_end = int(n_rows * 0.85)

df_train_raw = df_work.iloc[:train_end]
df_val_raw = df_work.iloc[train_end:val_end]
df_test_raw = df_work.iloc[val_end:]
print(f"[OK] Split: train={len(df_train_raw)}, val={len(df_val_raw)}, test={len(df_test_raw)}")

# ============================================================
# STEP 6 — NORMALISASI (scaler HANYA dari train)
# ============================================================
train_min = df_train_raw[ALL_COLS].min().values
train_max = df_train_raw[ALL_COLS].max().values
np.save(BASE_DIR / "scaler_params.npy", np.stack([train_min, train_max]))

def normalize(arr, mn, mx):
    return (arr - mn) / (mx - mn + 1e-8)

train_norm = normalize(df_train_raw[ALL_COLS].values, train_min, train_max)
val_norm = normalize(df_val_raw[ALL_COLS].values, train_min, train_max)
test_norm = normalize(df_test_raw[ALL_COLS].values, train_min, train_max)

# ============================================================
# STEP 7 — WINDOWING
# ============================================================
def make_windows(data_array, level_array, feature_count=5):
    X, y, y_level_last = [], [], []
    for i in range(len(data_array) - WINDOW_SIZE - HORIZON + 1):
        X.append(data_array[i:i + WINDOW_SIZE, :feature_count])
        y.append(data_array[i + WINDOW_SIZE - 1:i + WINDOW_SIZE - 1 + HORIZON, -1])
        y_level_last.append(level_array[i + WINDOW_SIZE - 1])
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(y_level_last, dtype=np.float32),
    )

X_train, y_train, _ = make_windows(train_norm, df_train_raw["load_output_level"].values)
X_val, y_val, _ = make_windows(val_norm, df_val_raw["load_output_level"].values)
X_test, y_test, y_test_level_last = make_windows(test_norm, df_test_raw["load_output_level"].values)

# ============================================================
# STEP 8 — SIMPAN
# ============================================================
np.save(BASE_DIR / "X_train.npy", X_train)
np.save(BASE_DIR / "y_train.npy", y_train)
np.save(BASE_DIR / "X_val.npy", X_val)
np.save(BASE_DIR / "y_val.npy", y_val)
np.save(BASE_DIR / "X_test.npy", X_test)
np.save(BASE_DIR / "y_test.npy", y_test)
np.save(BASE_DIR / "y_test_level_last.npy", y_test_level_last)

print(f"\n[OK] Windows: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
print(f"[SUCCESS] Pipeline selesai! Dataset: {resolution_label}")
if USE_SYNTHETIC:
    print(f"Window = {WINDOW_SIZE} menit histori ({WINDOW_SIZE} × 1 menit)")
else:
    print(f"Window = {WINDOW_SIZE * 15 / 60:.1f} jam histori ({WINDOW_SIZE} × 15 menit)")