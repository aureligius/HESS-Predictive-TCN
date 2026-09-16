# test_normalized_sum_tcn_arima.py
"""
Uji lanjutan: setelah SUM dinormalisasi per-sumber (linearitas turun
dari R²=0.55 ke R²=0.11 dengan Linear Regression), bagaimana performa
TCN dan ARIMA pada target BARU ini?

Kalau TCN > Linear Regression > ARIMA di sini: maka
TCN unggul saat sinyal genuinely non-linear.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
CSV_PATH = DATA_DIR / "sumber_energi_15.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

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

print(f"[OK] Preprocessing selesai. Shape: {df.shape}\n")

# STEP 2: NORMALISASI PER-SUMBER, LALU JUMLAHKAN
df_norm = df.copy()
for col in FEATURE_COLS:
    col_min, col_max = df[col].min(), df[col].max()
    df_norm[col] = (df[col] - col_min) / (col_max - col_min + 1e-8)

df["sum_normalized"] = df_norm[FEATURE_COLS].sum(axis=1)
df["sum_normalized_level"] = df["sum_normalized"]
df["sum_normalized_delta"] = df["sum_normalized"].diff().shift(-1)
df = df.iloc[:-1].reset_index(drop=True)

WINDOW_SIZE = 48

# STEP 3: Split + windowing UNTUK TCN (multivariat, 5 fitur ternormalisasi)
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_min = df.iloc[:train_end]["sum_normalized_delta"].min()
train_max = df.iloc[:train_end]["sum_normalized_delta"].max()

def norm_target(arr, mn, mx):
    return (arr - mn) / (mx - mn + 1e-8)

delta_norm_target = norm_target(df["sum_normalized_delta"].values, train_min, train_max)
X_features = df_norm[FEATURE_COLS].values[:-1]  # sinkron dengan df setelah iloc[:-1] di atas

def make_windows(X_feat, y_target, window_size):
    X, y = [], []
    for i in range(len(X_feat) - window_size):
        X.append(X_feat[i:i+window_size])
        y.append(y_target[i+window_size-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_windows(X_features, delta_norm_target, WINDOW_SIZE)
n_w = len(X_all)
tr_end_w = int(n_w * 0.70)
val_end_w = int(n_w * 0.85)

X_train, y_train = X_all[:tr_end_w], y_all[:tr_end_w]
X_val, y_val = X_all[tr_end_w:val_end_w], y_all[tr_end_w:val_end_w]
X_test, y_test = X_all[val_end_w:], y_all[val_end_w:]

print(f"[OK] Windows: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}\n")

# STEP 4: TCN MINI (arsitektur sama seperti final: 32-64-64)
class CausalConv1d(nn.Module):
    def __init__(self, ic, oc, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(ic, oc, k, padding=self.pad, dilation=d)
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.pad] if self.pad else x

class TCNBlock(nn.Module):
    def __init__(self, ic, oc, k, d, dr=0.2):
        super().__init__()
        self.c1 = CausalConv1d(ic, oc, k, d); self.r1 = nn.ReLU(); self.d1 = nn.Dropout(dr)
        self.c2 = CausalConv1d(oc, oc, k, d); self.r2 = nn.ReLU(); self.d2 = nn.Dropout(dr)
        self.ds = nn.Conv1d(ic, oc, 1) if ic != oc else None
        self.ro = nn.ReLU()
    def forward(self, x):
        out = self.d1(self.r1(self.c1(x)))
        out = self.d2(self.r2(self.c2(out)))
        res = x if self.ds is None else self.ds(x)
        return self.ro(out + res)

class TCNTest(nn.Module):
    def __init__(self, ni=5, ch=(32, 64, 64), k=3, dr=0.2):
        super().__init__()
        layers = []; ic = ni
        for i, oc in enumerate(ch):
            layers.append(TCNBlock(ic, oc, k, 2**i, dr)); ic = oc
        self.tcn = nn.Sequential(*layers)
        self.fc1 = nn.Linear(ch[-1], 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dr)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=128, shuffle=False)

model = TCNTest().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=5)
crit = nn.MSELoss()

best_val, best_state, patience_counter = float("inf"), None, 0
print("[INFO] Training TCN pada SUM ternormalisasi...")
for epoch in range(1, 101):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(Xb), yb)
        loss.backward()
        opt.step()
    model.eval()
    total_val = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            total_val += crit(model(Xb), yb).item() * Xb.size(0)
    val_loss = total_val / len(val_loader.dataset)
    sched.step(val_loss)
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 15:
            print(f"[INFO] Early stopping di epoch {epoch}")
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred_tcn = model(torch.from_numpy(X_test).to(device)).cpu().numpy().flatten()
r2_tcn = r2_score(y_test, pred_tcn)
print(f"[OK] TCN selesai. R² = {r2_tcn:.4f}\n")

# STEP 5: LINEAR REGRESSION (baseline pembanding, konfirmasi ulang)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
lr = LinearRegression()
lr.fit(X_train_flat, y_train)
pred_lr = lr.predict(X_test_flat)
r2_lr = r2_score(y_test, pred_lr)
print(f"[OK] Linear Regression: R² = {r2_lr:.4f}\n")

# STEP 6: ARIMA (univariat, dari delta_norm_target langsung)
delta_arima = df["sum_normalized_delta"].values
train_end_arima = int(len(delta_arima) * 0.70)
val_end_arima = int(len(delta_arima) * 0.85)

arima_train = delta_arima[:train_end_arima]
arima_test = delta_arima[val_end_arima:]

print("[INFO] Fitting ARIMA(2,0,2) pada SUM ternormalisasi...")
model_arima = ARIMA(arima_train, order=(2, 0, 2)).fit()

predictions = []
current_model = model_arima
for i in range(len(arima_test)):
    pred = current_model.forecast(steps=1)[0]
    predictions.append(pred)
    current_model = current_model.append([arima_test[i]], refit=False)

pred_arima = np.array(predictions)
r2_arima = r2_score(arima_test, pred_arima)
print(f"[OK] ARIMA selesai. R² = {r2_arima:.4f}\n")

# STEP 7: TABEL PERBANDINGAN FINAL
print("=" * 60)
print("   PERBANDINGAN PADA SUM TERNORMALISASI (dominasi solar dihilangkan)")
print("=" * 60)
print(f"{'Model':<20} {'R²':>10}")
print("-" * 60)
print(f"{'TCN':<20} {r2_tcn:>10.4f}")
print(f"{'Linear Regression':<20} {r2_lr:>10.4f}")
print(f"{'ARIMA(2,0,2)':<20} {r2_arima:>10.4f}")
print("=" * 60)
print("\n[REFERENSI] R² pada SUM MENTAH (dominasi solar utuh, dari Table 1):")
print("  TCN=0.4969, Linear Regression=0.5600, ARIMA=0.5258")