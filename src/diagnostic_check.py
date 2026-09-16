# diagnostic_check.py
"""
Diagnostik: mengecek apakah delta target kita banyak berisi nilai NOL PERSIS
akibat forward-fill dari data hourly RTE yang dipaksa jadi 15-menit.

Kalau proporsi delta=0 tinggi dan pola run-nya teratur, ini indikasi bahwa
fluktuasi 15 menit kita sebagian besar artefak dan bukan pengukuran asli 
yang bisa menjelaskan kenapa model linear (ARIMA, Linear Regression)
mengungguli TCN.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
CSV_PATH = DATA_DIR / "sumber_energi_15.csv"

# STEP 1: Preprocessing sampai delta 
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

# STEP 2: CEK 1: Proporsi delta = 0 PERSIS
delta = df["load_output"]

zero_count = (delta == 0).sum()
near_zero_count = (delta.abs() < 1.0).sum()
total = len(delta)

print("=" * 60)
print("   CEK 1 — PROPORSI DELTA NOL")
print("=" * 60)
print(f"Total baris delta         : {total}")
print(f"Delta PERSIS nol          : {zero_count} ({zero_count/total*100:.1f}%)")
print(f"Delta hampir nol (<1 MW)  : {near_zero_count} ({near_zero_count/total*100:.1f}%)")
print()

# STEP 3: CEK 2: Panjang run delta=0 berturut-turut
is_zero = (delta == 0).astype(int)
# Kelompokkan baris berturut-turut yang statusnya sama (0 atau bukan-0)
group_id = (is_zero != is_zero.shift()).cumsum()
run_lengths = is_zero.groupby(group_id).sum()
zero_runs = run_lengths[run_lengths > 0]

print("=" * 60)
print("   CEK 2 — POLA RUN 'DELTA=0 BERTURUT-TURUT'")
print("=" * 60)
print(f"Jumlah 'run' delta=0 yang terjadi : {len(zero_runs)}")
print(f"Rata-rata panjang run             : {zero_runs.mean():.2f} interval")
print(f"Run terpanjang                    : {zero_runs.max()} interval")
print(f"\nDistribusi panjang run (top 10 paling sering):")
print(zero_runs.value_counts().sort_index().head(10))
print()

# STEP 4: CEK 3: Per-kolom, seberapa sering NILAI (bukan delta) berulang
#          persis 2-3x berturut-turut (indikasi langsung forward-fill)
print("=" * 60)
print("   CEK 3 — PENGULANGAN NILAI PERSIS PER KOLOM (indikasi forward-fill)")
print("=" * 60)
for col in FEATURE_COLS:
    vals = df[col]
    is_same_as_prev = (vals == vals.shift()).astype(int)
    pct_repeated = is_same_as_prev.mean() * 100
    print(f"  {col:12s}: {pct_repeated:5.1f}% baris nilainya SAMA PERSIS dengan baris sebelumnya")
print()

# STEP 5: KESIMPULAN 
print("=" * 60)
print("   INTERPRETASI")
print("=" * 60)
zero_pct = zero_count / total * 100
if zero_pct > 30:
    print(f"[TEMUAN] {zero_pct:.1f}% delta = 0 PERSIS — proporsi tinggi.")
    print("Ini mengindikasikan sebagian besar 'fluktuasi 15-menit' kita")
    print("kemungkinan ARTIFAK forward-fill (nilai diulang dari sumber per-jam),")
    print("bukan pengukuran independen asli tiap 15 menit.")
    print("\nIMPLIKASI: data per-menit ASLI (dari prototype fisik nanti)")
    print("kemungkinan akan menunjukkan fluktuasi yang LEBIH genuine dan")
    print("lebih non-linear — di mana keunggulan TCN atas model linear")
    print("(ARIMA, Linear Regression) berpotensi lebih terlihat.")
elif zero_pct > 10:
    print(f"[TEMUAN] {zero_pct:.1f}% delta = 0 PERSIS — proporsi sedang.")
    print("Sebagian data kemungkinan artifak forward-fill, tapi bukan mayoritas.")
    print("Perlu investigasi lebih lanjut sebelum menyimpulkan penyebab")
    print("dominasi model linear atas TCN.")
else:
    print(f"[TEMUAN] {zero_pct:.1f}% delta = 0 PERSIS — proporsi rendah.")
    print("Forward-fill KEMUNGKINAN BESAR BUKAN penyebab utama dominasi")
    print("model linear. Perlu dicari penjelasan lain (misal: data memang")
    print("secara inheren didominasi struktur linear/autokorelasi kuat).")
print("=" * 60)