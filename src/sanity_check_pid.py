# sanity_check_pid.py
"""
Ganti prediksi TCN dengan angka RANDOM MURNI.
Kalau PID reduction MASIH tinggi, itu bukti metrik PID reduction kita
tidak sensitif terhadap kualitas prediksi tapi dominan dari struktur
setpoint smoothing, bukan skill model.
"""
from pathlib import Path
import numpy as np
from show import build_results

results = build_results(plot=False)
level_actual_next = results["level_actual_next"]

# GANTI prediksi dengan RANDOM murni (skala sama dengan level aktual)
np.random.seed(42)
level_pred_random = level_actual_next + np.random.normal(
    0, level_actual_next.std() * 0.5, len(level_actual_next)
)

WINDOW = 96
setpoint = np.convolve(level_actual_next, np.ones(WINDOW) / WINDOW, mode="same")

Kp, Ki, Kd = 0.7, 0.05, 0.1
INTEGRAL_CLAMP = 500
integral, prev_error = 0.0, 0.0
u = np.zeros(len(level_actual_next))

for t in range(len(level_actual_next)):
    error = level_pred_random[t] - setpoint[t]   # pakai prediksi RANDOM
    integral = np.clip(integral + error, -INTEGRAL_CLAMP, INTEGRAL_CLAMP)
    derivative = error - prev_error
    u[t] = Kp * error + Ki * integral + Kd * derivative
    prev_error = error

smoothed = level_actual_next - u
dev_raw = np.mean(np.abs(level_actual_next - setpoint))
dev_smoothed = np.mean(np.abs(smoothed - setpoint))
reduction_pct = (1 - dev_smoothed / dev_raw) * 100

print(f"PID Reduction dengan prediksi RANDOM MURNI: {reduction_pct:.1f}%")
print("\n[INTERPRETASI]")
print("Kalau angka ini juga TINGGI (mendekati 70%), berarti PID reduction")
print("kita TIDAK sensitif terhadap kualitas prediksi — perlu redesign")
print("metrik evaluasi PID atau WINDOW setpoint.")