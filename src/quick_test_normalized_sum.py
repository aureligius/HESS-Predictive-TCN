# quick_test_normalized_sum.py
"""
Uji cepat: apakah menjumlahkan sumber yang SUDAH dinormalisasi (skala 0-1)
mengubah linearitas SUM dibanding sum dari nilai mentah?
Tidak retrain TCN — cuma cek linear regression pada target BARU.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
CSV_PATH = DATA_DIR / "sumber_energi_15.csv"

# STEP 1: Preprocessing
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

print(f"[OK] Preprocessing selesai. Shape: {df.shape}\n")

# NORMALISASI TIAP SUMBER SECARA TERPISAH (0-1) SEBELUM DIJUMLAHKAN
FEATURE_COLS = ["wind", "thermal", "geothermal", "hydrogen", "wave"]
df_norm = df.copy()
for col in FEATURE_COLS:
    col_min, col_max = df[col].min(), df[col].max()
    df_norm[col] = (df[col] - col_min) / (col_max - col_min + 1e-8)

# SUM dari versi ternormalisasi (bukan MW mentah)
df_norm["sum_normalized"] = df_norm[FEATURE_COLS].sum(axis=1)
delta_normalized_sum = df_norm["sum_normalized"].diff().shift(-1).dropna()

# UJI: R² Linear Regression pada delta SUM ternormalisasi
def make_windows_1d(series, window_size=48):
    X, y = [], []
    vals = series.values
    for i in range(len(vals) - window_size - 1):
        X.append(vals[i:i+window_size])
        y.append(vals[i+window_size])
    return np.array(X), np.array(y)

X, y = make_windows_1d(delta_normalized_sum, window_size=48)
split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
r2_normalized = r2_score(y_test, pred)

print(f"R² Linear Regression pada SUM ternormalisasi: {r2_normalized:.4f}")
print(f"(Bandingkan dengan R² pada SUM mentah sebelumnya: 0.5467)")
print()
if r2_normalized < 0.50:
    print("[TEMUAN] R² turun setelah normalisasi per-sumber.")
    print("Ini mengkonfirmasi: dominasi solar SEBAGIAN disebabkan oleh")
    print("magnitude MW-nya yang besar, bukan cuma pola periodiknya.")
else:
    print("[TEMUAN] R² tidak banyak berubah.")
    print("Dominasi solar lebih disebabkan POLA periodiknya sendiri,")
    print("bukan sekadar besarnya magnitude MW.")