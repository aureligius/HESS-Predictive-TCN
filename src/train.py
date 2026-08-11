import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model import HESS_TCN_v2

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)  

torch.manual_seed(42) # biar random number yang di generate itu punya pola yang sama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WindowDataset(Dataset):
    def __init__(self, X_path, y_path): # ubah data jadi tensor PyTorch
        self.X = torch.from_numpy(np.load(X_path))
        self.y = torch.from_numpy(np.load(y_path))
    def __len__(self): return len(self.X) # panjang row dari set yang dipakai
    def __getitem__(self, idx): return self.X[idx], self.y[idx] # berikan sepasang soal + jawaban

BATCH_SIZE = 128 # 1 epoch = 128 set
train_loader = DataLoader(
    WindowDataset(OUTPUT_DIR / "X_train.npy", OUTPUT_DIR / "y_train.npy"),
    batch_size=BATCH_SIZE,
    shuffle=True,
)  # acak urutan tiap epoch, biar model gak menghafal urutan
val_loader = DataLoader(
    WindowDataset(OUTPUT_DIR / "X_val.npy", OUTPUT_DIR / "y_val.npy"),
    batch_size=BATCH_SIZE,
    shuffle=False,
)  # untuk cek performa, jadi tidak perlu shuffle

# KUNCI UTAMA: num_inputs diset tepat ke angka 5!
model = HESS_TCN_v2(num_inputs=5).to(device)

EPOCHS = 100
PATIENCE = 15
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=5)

best_val_loss = float("inf")
patience_counter = 0
train_losses, val_losses = [], []

print("[INFO] Memulai training dengan 5 Fitur Mikro-Grid Hijau...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * X_b.size(0)
    train_loss = total_train_loss / len(train_loader.dataset)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            total_val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
    val_loss = total_val_loss / len(val_loader.dataset)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"[INFO] Early stopping aktif di epoch {epoch}.")
            break

    train_losses.append(train_loss)
    val_losses.append(val_loss)


np.save(OUTPUT_DIR / "loss_history.npy", np.array([train_losses, val_losses]))
print(f"[SUCCESS] Training Selesai! Best Val Loss: {best_val_loss:.6f}")