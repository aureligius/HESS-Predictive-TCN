# check_bias.py
"""
Cek kecurigaan: apakah directional accuracy tinggi pada data Brownian Bridge
itu genuine skill model, atau "gratis" dari bias struktural cara kita
membuat data sintetis (karena tiap blok 15-menit DIPAKSA berakhir di titik
15-menit asli, sehingga arah menit-per-menit di dalamnya cenderung searah
dengan arah blok besarnya).

File ini BERDIRI SENDIRI — cuma baca ulang X_train/X_test yang SEDANG ada
di output/ (asumsi masih berisi data Brownian Bridge dari eksperimen
terakhir). Tidak menyentuh pipeline lain.
"""
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "output"

# ============================================================
# Load delta AKTUAL dari test set (bukan dari df_work langsung,
# supaya konsisten dengan apa yang benar-benar dievaluasi show.py)
# ============================================================
y_test_norm = np.load(OUTPUT_DIR / "y_test.npy").flatten()
scaler = np.load(OUTPUT_DIR / "scaler_params.npy")
target_min, target_max = scaler[0, -1], scaler[1, -1]

def denorm(arr, mn, mx):
    return arr * (mx - mn) + mn

delta_minute = denorm(y_test_norm, target_min, target_max)

print(f"[OK] Total delta per-menit di test set: {len(delta_minute)}")

# ============================================================
# Kelompokkan per blok 15 menit, cek alignment arah
# ============================================================
n_blocks = len(delta_minute) // 15
same_direction_scores = []

for i in range(n_blocks):
    block = delta_minute[i*15 : (i+1)*15]
    if len(block) < 15:
        continue
    block_sum_sign = np.sign(block.sum())
    if block_sum_sign == 0:
        continue  # skip blok yang totalnya persis nol (jarang, tapi jaga-jaga)
    per_minute_signs = np.sign(block)
    match_ratio = (per_minute_signs == block_sum_sign).mean()
    same_direction_scores.append(match_ratio)

avg_alignment = np.mean(same_direction_scores)
median_alignment = np.median(same_direction_scores)

print("\n" + "=" * 60)
print("   CEK BIAS STRUKTURAL BROWNIAN BRIDGE")
print("=" * 60)
print(f"Jumlah blok 15-menit dianalisis : {len(same_direction_scores)}")
print(f"Rata-rata alignment             : {avg_alignment:.3f}")
print(f"Median alignment                : {median_alignment:.3f}")
print()

# ============================================================
# Interpretasi otomatis
# ============================================================
if avg_alignment > 0.70:
    print("[TEMUAN] Alignment TINGGI (>70%).")
    print("Ini mengindikasikan sebagian besar arah delta per-menit memang")
    print("SEARAH dengan arah blok 15-menit besarnya — kemungkinan BESAR")
    print("directional accuracy 83% yang kita dapat itu SEBAGIAN 'GRATIS'")
    print("dari struktur Brownian Bridge (dipaksa mendarat di titik akhir),")
    print("bukan murni skill model menangkap fluktuasi genuine per-menit.")
elif avg_alignment > 0.55:
    print("[TEMUAN] Alignment SEDANG (55-70%).")
    print("Ada bias struktural, tapi tidak dominan. Directional accuracy")
    print("kemungkinan sebagian genuine skill, sebagian bias struktural.")
else:
    print("[TEMUAN] Alignment RENDAH (<55%, mendekati acak/50%).")
    print("Arah menit-per-menit TIDAK secara sistematis mengikuti arah blok")
    print("15-menit — directional accuracy 83% kemungkinan besar GENUINE")
    print("skill model, bukan artifak Brownian Bridge.")
print("=" * 60)