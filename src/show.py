from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import HESS_TCN_v2   

BASE_DIR = Path(__file__).resolve().parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_results(plot=True):
    model = HESS_TCN_v2(num_inputs=5).to(device)
    model.load_state_dict(torch.load(BASE_DIR / "best_model.pth", map_location=device))
    model.eval()

    # 1. LOAD DATA TEST, SCALER, DAN LEVEL TERAKHIR (untuk rekonstruksi)
    X_test = np.load(BASE_DIR / "X_test.npy")
    y_test = np.load(BASE_DIR / "y_test.npy")
    y_test_level_last = np.load(BASE_DIR / "y_test_level_last.npy")
    scaler = np.load(BASE_DIR / "scaler_params.npy")
    target_min, target_max = scaler[0, -1], scaler[1, -1]

    # 2. PREDIKSI (di GPU, lalu pindah balik ke CPU)
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test).to(device)
        preds_norm = model(X_tensor).cpu().numpy()

    def denorm(arr, mn, mx):
        return arr * (mx - mn) + mn

    # 3. DENORMALISASI KE SKALA DELTA ASLI (MW)
    delta_actual = denorm(y_test, target_min, target_max)
    delta_pred = denorm(preds_norm, target_min, target_max)

    # 4. METRIK PADA DELTA -> ini ukuran SKILL ASLI model menangkap fluktuasi
    r2_delta = r2_score(delta_actual, delta_pred)
    mae_delta = mean_absolute_error(delta_actual, delta_pred)

    # 5. REKONSTRUKSI KE LEVEL (MW) -> ini yang dipakai untuk grafik "Beban Aktual vs Prediksi"
    level_actual_next = y_test_level_last + delta_actual.flatten()
    level_pred_recon = y_test_level_last + delta_pred.flatten()

    r2_level = r2_score(level_actual_next, level_pred_recon)
    mae_level = mean_absolute_error(level_actual_next, level_pred_recon)
    rmse_level = np.sqrt(mean_squared_error(level_actual_next, level_pred_recon))

    # 6. PRINT HASIL — DUA METRIK, JANGAN CAMPUR
    print("\n" + "=" * 55)
    print("   HASIL EVALUASI MODEL TCN_v2 (DELTA-TARGET)")
    print("=" * 55)
    print(f"  R2 pada DELTA (skill asli AI)         : {r2_delta:.4f}")
    print(f"  MAE pada DELTA                         : {mae_delta:.2f} MW")
    print()
    print(f"  R2 pada LEVEL (hasil rekonstruksi)      : {r2_level:.4f}")
    print(f"  MAE pada LEVEL (hasil rekonstruksi)     : {mae_level:.2f} MW")
    print(f"  RMSE pada LEVEL (hasil rekonstruksi)    : {rmse_level:.2f} MW")
    print("=" * 55 + "\n")

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(level_actual_next, label="Actual Load", linewidth=1.2, alpha=0.8)
        plt.plot(level_pred_recon, label="Predicted Load (HESS-TCN v2, delta-reconstructed)", linewidth=1.2, alpha=0.8)
        plt.xlabel("Time Step (hours)")
        plt.ylabel("Load Output (MW)")
        plt.title(f"Predicted vs Actual Load (Delta Reconstruction) | R2={r2_level:.3f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(BASE_DIR / "fig_pred_vs_actual.png", dpi=150)
        plt.show()
        plt.close()
        print(f"[OK] Grafik tersimpan -> {BASE_DIR / 'fig_pred_vs_actual.png'}")

    # Tambahkan di show.py dalam build_results(), setelah plot prediksi vs aktual
    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        lim = max(np.abs(delta_actual).max(), np.abs(delta_pred).max())
        ax.scatter(delta_actual.flatten(), delta_pred.flatten(),
                alpha=0.15, s=3, color="#534AB7")
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.5,
                label="Prediksi sempurna")
        ax.set_xlabel("Delta Aktual (MW)")
        ax.set_ylabel("Delta Prediksi (MW)")
        ax.set_title(f"Scatter: Prediksi vs Aktual Delta\nR²={r2_delta:.4f}")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(BASE_DIR / "fig_scatter_delta.png", dpi=150)
        plt.close()

    return {
        "level_actual_next": level_actual_next,
        "level_pred_recon": level_pred_recon,
        "delta_actual": delta_actual,
        "delta_pred": delta_pred,
        "r2_delta": r2_delta,
        "mae_delta": mae_delta,
        "r2_level": r2_level,
        "mae_level": mae_level,
        "rmse_level": rmse_level,
    }


if __name__ == "__main__":
    build_results(plot=True)
