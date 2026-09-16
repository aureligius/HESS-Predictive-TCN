import numpy as np
import matplotlib.pyplot as plt
from show import build_results
from config import USE_SYNTHETIC
from pathlib import Path
from pandas import pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)  

results = build_results(plot=False)
level_actual_next = results["level_actual_next"]
level_pred_recon = results["level_pred_recon"]

# 1. SETPOINT: target stabil = rolling average dari beban aktual
# (baseline daya aman yang ingin dijaga konstan ke BESS)
if USE_SYNTHETIC:
    WINDOW = 1440
else:
    WINDOW = 96
setpoint = pd.Series(level_actual_next).rolling(WINDOW, min_periods=1).mean().values

# 2. KONSTANTA PID (ILUSTRATIF dan akan dikalibrasi ulang saat implementasi hardware)
Kp, Ki, Kd = 0.7, 0.05, 0.1
INTEGRAL_CLAMP = 500  # anti-windup, supaya komponen integral tidak meledak

integral = 0.0
prev_error = 0.0
u = np.zeros(len(level_actual_next))

for t in range(len(level_actual_next)):
    # Error dihitung dari PREDIKSI AI terhadap setpoint (feedforward, sesuai arsitektur kita)
    error = level_pred_recon[t] - setpoint[t]
    integral += error
    integral = np.clip(integral, -INTEGRAL_CLAMP, INTEGRAL_CLAMP)
    derivative = error - prev_error
    u[t] = Kp * error + Ki * integral + Kd * derivative
    prev_error = error

# 3. SINYAL AKHIR YANG MASUK KE BESS SETELAH DIKOREKSI FLYWHEEL
smoothed_output = level_actual_next - u

# 4. UKUR SEBERAPA BESAR FLUKTUASI BERHASIL DIREDAM
dev_raw = np.mean(np.abs(level_actual_next - setpoint))
dev_smoothed = np.mean(np.abs(smoothed_output - setpoint))
reduction_pct = (1 - dev_smoothed / dev_raw) * 100

print(f"Deviasi rata-rata TANPA kontrol  : {dev_raw:.2f} MW")
print(f"Deviasi rata-rata DENGAN AI+PID  : {dev_smoothed:.2f} MW")
print(f"Reduksi fluktuasi                : {reduction_pct:.1f}%")

# 5. GRAFIK DEMONSTRASI UNTUK PROPOSAL
plt.figure(figsize=(12, 5))
plt.plot(level_actual_next[:300], label="Without Control (raw)", linewidth=1, alpha=0.6)
plt.plot(setpoint[:300], label="Setpoint (stable target)", linewidth=1.5, linestyle="--", color="black")
plt.plot(smoothed_output[:300], label="With AI + Predictive PID", linewidth=1.5)
plt.xlabel("Time Step (hours)")
plt.ylabel("Daya ke BESS (MW)")
plt.title(f"Fluctuation Damping Simulation: AI+PID vs No Control (reduction {reduction_pct:.1f}%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_pid_simulation.png", dpi=150)
plt.show()
plt.close()
print("[OK] Grafik tersimpan -> fig_pid_simulation.png")