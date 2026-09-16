# hardware_profile.py
"""
Export model ke ONNX dan profile ke Edge Impulse (espressif-esp32)
Run dari folder src/
"""
from pathlib import Path
import torch
from model import HESS_TCN_v2

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STEP 1:  Muat model yang sudah dilatih
model = HESS_TCN_v2(num_inputs=5).to(device)
model.load_state_dict(
    torch.load(OUTPUT_DIR / "best_model.pth", map_location=device, weights_only=True)
)
model.eval()
print("[OK] Model dimuat dari best_model.pth")

# STEP 2: Export ke ONNX
dummy_input = torch.randn(1, 48, 5).to(device)   # (batch, window=48, fitur=5)
onnx_path = OUTPUT_DIR / "hess_tcn_v2.onnx"

torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamo=False,
)
print(f"[OK] Model di-export ke {onnx_path}")

# STEP 3: Profile ke Edge Impulse
import edgeimpulse as ei

# GANTI dengan API Key (role harus 'admin')
ei.API_KEY = "ei_47cdc8244578bb881cbdc1efb30c8f0d550fa45786305abb"

print("\n[INFO] Menghubungi Edge Impulse untuk profiling...")
result = ei.model.profile(model=str(onnx_path), device="espressif-esp32")
print(result.summary())