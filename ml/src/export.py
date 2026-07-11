"""Export the trained scikit-learn model to ONNX for use in the extension.

Standalone: loads the model and feature order saved by train.py, converts to
ONNX, and copies both artifacts into extension/model/.
"""

import json
import pickle
import shutil
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
EXT_MODEL_DIR = ROOT.parent / "extension" / "model"

# Load the trained model and the feature column order.
with open(MODEL_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(MODEL_DIR / "feature_names.json") as f:
    feature_names = json.load(f)
n_features = len(feature_names)

# Convert to ONNX. zipmap=False makes predict_proba emit a plain float array
# ([[p0, p1], ...]) instead of a list of {label: prob} dicts, which is far
# easier to read from onnxruntime-web in the extension.
initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={id(model): {"zipmap": False}},
    target_opset=17,
)

onnx_path = MODEL_DIR / "clarity_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

size_kb = round(len(onnx_model.SerializeToString()) / 1024)
print(f"Exported {onnx_path}  ({size_kb} KB)")
if size_kb > 500:
    print("  WARNING: over 500 KB — consider fewer trees / lower max_depth.")

# Copy artifacts into the extension so it can load them at runtime.
EXT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(onnx_path, EXT_MODEL_DIR / "clarity_model.onnx")
shutil.copy(MODEL_DIR / "feature_names.json",
            EXT_MODEL_DIR / "feature_names.json")
print(f"Copied model + feature_names.json to {EXT_MODEL_DIR}")
