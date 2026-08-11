import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from model import HESS_TCN_v2
import matplotlib.colors as mcolors

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)  

# Load data
y_true_norm = np.load(OUTPUT_DIR / "y_test.npy").flatten()
scaler = np.load(OUTPUT_DIR / "scaler_params.npy")
target_min, target_max = scaler[0, -1], scaler[1, -1]

# Denormalisasi
def denorm(arr, mn, mx): return arr * (mx - mn) + mn

# Prediksi ulang dari model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HESS_TCN_v2(num_inputs=5).to(device)
model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth", map_location=device))
model.eval()
X_test = np.load(OUTPUT_DIR / "X_test.npy")
with torch.no_grad():
    preds_norm = model(torch.from_numpy(X_test).to(device)).cpu().numpy().flatten()

delta_actual = denorm(y_true_norm, target_min, target_max)
delta_pred   = denorm(preds_norm,  target_min, target_max)

# KONVERSI KE KLASIFIKASI ARAH
# 1 = naik (delta > 0), 0 = turun (delta <= 0)
# Filter: buang kasus delta sangat kecil (< 50 MW) — terlalu noise untuk dinilai arahnya
THRESHOLD = 50  # MW
mask = np.abs(delta_actual) > THRESHOLD

actual_dir = (delta_actual[mask] > 0).astype(int)
pred_dir   = (delta_pred[mask]   > 0).astype(int)

print("="*55)
print("   METRIK ARAH PREDIKSI (|delta| > 50 MW)")
print("="*55)
print(f"Jumlah sampel (fluktuasi signifikan): {mask.sum()}")
print(f"Directional Accuracy: {(actual_dir == pred_dir).mean():.4f}")
print()
print(classification_report(actual_dir, pred_dir,
      target_names=["Turun (delta<0)", "Naik (delta>0)"]))

# CONFUSION MATRIX
cm = confusion_matrix(actual_dir, pred_dir)
fig, ax = plt.subplots(figsize=(6, 5))

# Buat color matrix: diagonal=1 (biru tua), sisanya=0 (putih)
color_matrix = np.eye(cm.shape[0])
cmap = mcolors.LinearSegmentedColormap.from_list(
    'custom', ['white', '#0A2972'])
ax.imshow(color_matrix, cmap=cmap, vmin=0, vmax=1)

labels_cm = ["DECREASE", "INCREASE"]
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center', fontsize=16,
                fontweight='bold',
                color='white' if i == j else 'black')

ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(labels_cm); ax.set_yticklabels(labels_cm)
ax.set_xlabel("PREDICTION", fontsize=12)
ax.set_ylabel("ACTUAL", fontsize=12)
ax.set_title("Confusion Matrix — Predicting Delta Fluctuations\n"
             "(Blue = True, White = False)", fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# GRAFIK LOSS CURVE (train vs val)
try:
    history = np.load(OUTPUT_DIR / "loss_history.npy")
    train_losses_arr = history[0]
    val_losses_arr   = history[1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses_arr, label="Train Loss", linewidth=1.5)
    ax.plot(val_losses_arr,   label="Val Loss",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss Curve: Train vs Validation")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_loss_curve.png", dpi=150)
    plt.show()
    plt.close()
    print("[OK] Loss curve tersimpan")
except FileNotFoundError:
    print("[INFO] loss_history.npy tidak ada")