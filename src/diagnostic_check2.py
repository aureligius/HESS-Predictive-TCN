# diagnostic_check2.py
"""
Uji hipotesis: apakah TCN unggul pada sumber INDIVIDUAL (sebelum dijumlahkan),
tapi kalah setelah dijumlahkan jadi SUM (karena efek averaging membuat
sinyal SUM lebih linear)?

Membandingkan R² Linear Regression vs korelasi-diri (autokorelasi) pada
level WIND SENDIRI (paling volatile, kontribusi ~46% variansi delta SUM).
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

def make_windows_1d(series, window_size=48):
    X, y = [], []
    vals = series.values
    for i in range(len(vals) - window_size - 1):
        X.append(vals[i:i+window_size])
        y.append(vals[i+window_size] - vals[i+window_size-1])  # delta
    return np.array(X), np.array(y)

# Split 70/15/15 sederhana untuk uji cepat
n = len(df)
train_end = int(n * 0.70)

results = {}
for col in FEATURE_COLS + ["load_output_level"]:
    series = df[col]
    X, y = make_windows_1d(series, window_size=48)
    split = int(len(X) * 0.85)  # pakai 85% pertama utk train, sisanya test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    r2 = r2_score(y_test, pred)
    results[col] = r2

print("=" * 60)
print("  R² LINEAR REGRESSION PER SUMBER INDIVIDUAL vs SUM")
print("=" * 60)
for col, r2 in results.items():
    print(f"  {col:20s}: R² = {r2:.4f}")
print("=" * 60)
print("\n[INTERPRETASI]")
print("Kalau R² pada sumber individual (wind, wave) jauh LEBIH RENDAH")
print("dari R² pada 'load_output_level' (SUM), ini konfirmasi bahwa")
print("proses PENJUMLAHAN membuat sinyal lebih mudah diprediksi secara")
print("linear — dan TCN berpotensi unggul pada sumber individual yang")
print("lebih non-linear, meski kalah saat semuanya dijumlahkan.")