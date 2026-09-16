# diagnostic_check3.py
"""
Uji langsung: apakah TCN mengungguli Linear Regression pada sumber
INDIVIDUAL yang non-periodik (wave, wind) dibanding pada SUM
yang affected solar linearity 

Melatih TCN KECIL (ringan, cepat) khusus untuk memprediksi delta
WAVE saja (1 fitur input, bukan 5), dibandingkan Linear Regression
pada target yang sama.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

WINDOW_SIZE = 48
TARGET_SOURCE = "wave"   # ganti ke "wind" untuk uji sumber lain

# STEP 2: Buat delta + windows KHUSUS untuk 1 kolom saja
series = df[TARGET_SOURCE].copy()
delta_series = series.diff().shift(-1)
level_series = series.copy()

valid = delta_series.notna()
delta_series = delta_series[valid].reset_index(drop=True)
level_series = level_series[valid].reset_index(drop=True)

n = len(delta_series)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_min, train_max = level_series.iloc[:train_end].min(), level_series.iloc[:train_end].max()
delta_min = delta_series.iloc[:train_end].min()
delta_max = delta_series.iloc[:train_end].max()

def normalize(arr, mn, mx):
    return (arr - mn) / (mx - mn + 1e-8)

level_norm = normalize(level_series.values, train_min, train_max)
delta_norm = normalize(delta_series.values, delta_min, delta_max)

def make_windows_1feat(level_arr, delta_arr, window_size):
    X, y = [], []
    for i in range(len(level_arr) - window_size):
        X.append(level_arr[i:i+window_size])
        y.append(delta_arr[i+window_size-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_windows_1feat(level_norm, delta_norm, WINDOW_SIZE)
# reindex train/val/test sesuai window
n_windows = len(X_all)
tr_end_w = int(n_windows * 0.70)
val_end_w = int(n_windows * 0.85)

X_train, y_train = X_all[:tr_end_w], y_all[:tr_end_w]
X_val, y_val = X_all[tr_end_w:val_end_w], y_all[tr_end_w:val_end_w]
X_test, y_test = X_all[val_end_w:], y_all[val_end_w:]

print(f"[OK] Target: {TARGET_SOURCE} | Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}\n")

# STEP 3: Linear Regression baseline
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, pred_lr)
print(f"[Linear Regression] R² pada delta {TARGET_SOURCE}: {r2_lr:.4f}")

# STEP 4: TCN KECIL (1 fitur input, arsitektur ringan)
class CausalConv1d(nn.Module):
    def __init__(self, ic, oc, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(ic, oc, k, padding=self.pad, dilation=d)
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.pad] if self.pad else x

class TCNBlockMini(nn.Module):
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

class TCNMini(nn.Module):
    def __init__(self, ni=1, ch=(16, 32, 32), k=3, dr=0.2):
        super().__init__()
        layers = []; ic = ni
        for i, oc in enumerate(ch):
            layers.append(TCNBlockMini(ic, oc, k, 2**i, dr)); ic = oc
        self.tcn = nn.Sequential(*layers)
        self.fc1 = nn.Linear(ch[-1], 32)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dr)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, window) -> (batch, 1, window) — 1 channel
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

model = TCNMini().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=5)
crit = nn.MSELoss()

best_val, best_state, patience_counter = float("inf"), None, 0
print(f"\n[INFO] Training TCN mini untuk {TARGET_SOURCE}...")
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

# STEP 5: Perbandingan Final
print("\n" + "=" * 60)
print(f"   PERBANDINGAN: TCN vs LINEAR REGRESSION pada '{TARGET_SOURCE}'")
print("=" * 60)
print(f"  Linear Regression R² : {r2_lr:.4f}")
print(f"  TCN (mini)        R² : {r2_tcn:.4f}")
print("=" * 60)
if r2_tcn > r2_lr:
    print(f"[TEMUAN] TCN UNGGUL pada sumber individual '{TARGET_SOURCE}'.")
    print("Ini mendukung argumen: TCN lebih baik pada komponen non-periodik,")
    print("meski kalah pada SUM yang didominasi solar yang sangat linear.")
else:
    print(f"[TEMUAN] Linear Regression tetap unggul bahkan pada '{TARGET_SOURCE}'.")
    print("Perlu revisi hipotesis lebih lanjut soal keunggulan TCN.")
print("=" * 60)